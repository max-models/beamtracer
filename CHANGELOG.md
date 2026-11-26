# Changelog

All notable changes to this project are documented in this file and maintained manually. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- wellbalanced is now carafe: the `neutralizer` extra installs `plasma-carafe[cr]`, and
  `beamtracer.neutralizer` imports `carafe`.

### Added

- Tabulated faces: `FaceProfile` (`FieldBoundaryCondition.profile(coordinate, points,
  values)`) is a Dirichlet value given at points along one coordinate of the face, linear
  between them; `{profile: {coordinate, points, values, unit}}` in a configuration file. It
  starts the beam optics on the potential another code computed on a plane, e.g. picnic's
  PIC of the extraction region (`picnic.coupling.BeamOpticsCoupling`).
- Relativistic tracing: `BeamTracer(relativistic=True)`, `BeamTracer3D(relativistic=True)`
  and `relativistic: true` in a `BeamOpticsConfig` move the particles by the relativistic
  equations of motion. The tracers advance the proper velocity `u = gamma v` (made from the
  particles' velocities at the start): the electric field changes it at `q E / m`, the
  magnetic force is `q u x B / gamma`, the positions, the Hermite midpoint of the deposit,
  the step estimate and the charge-changing path use `u / gamma`, and the plane crossings
  record `u / gamma`. Runge--Kutta integrates `du/dt` and `dx/dt = u / gamma`; Boris's step
  is mpiparticles' relativistic one (`boris_velocity(..., relativistic=True)`, and in the 3D
  kernel the block `relativistic_boris_update` / `mpiparticles_relativistic_boris`). The
  kernels `trace_trajectories_2d` and `trace_trajectories_3d` (Pyccel and CUDA) take
  `inv_c2` (`1 / c^2`, 0 for classical, which leaves their arithmetic unchanged) and use
  mpiparticles' `lorentz_factor` blocks; the 3D kernel's Boris half steps are mpiparticles'
  block in both modes. `beam_with_energy` and `beam_on_surface` take `relativistic`: the
  energy gives the proper speed `c sqrt(gamma^2 - 1)`, the spreads are drawn in `u`, and
  `J / v` uses `u / gamma`. Tests against hyperbolic motion (the speed and the deposited
  flight time) and the relativistic gyroradius in both tracers, compiled against NumPy, all
  kernel versions in both modes, and slow protons against the classical tracer. Example
  script "Relativistic electrons": 1 MV over 2 cm and a 1 MeV electron in 0.1 T.
- A graded beam axis: `GridParams(x_spacing=[(x, dx), ...])` gives the node spacing wanted
  along `x` (linear between the pairs), and `x_through` positions that must be nodes (the
  electrodes' faces and tips); the nodes come from `patchwork.graded_nodes`.
  `IonOpticsMesh(x_nodes=...)` takes them: the electrode distances are fractions of each
  edge, `StaggeredElectricField` differences over the physical edges, the field is solved by
  patchwork's `NodalPoisson` on a graded axis, refinement boxes split the graded cells evenly,
  and the space charge is divided by the physical node volumes. Both tracers locate points by
  the uniform coordinate of the graded axis (`IonOpticsMesh.logical_x`); the compiled kernel
  (Pyccel and CUDA) takes the axis's nodes and a flag, and a step the shorter of the patch's
  spacing and the local edge. Uniform meshes run as before. 2D only, not with adaptive
  refinement. Example script "Graded beam axis": an extraction and a 120 mm drift with the beam
  of a uniform 0.25 mm mesh on 40 % of its nodes, and with refinement boxes over the gap and
  along the beam a closer beam on 30 % of them. Example notebook "Graded mesh": a cylindrical
  extraction and drift with the divergence of a uniform 0.05 mm mesh on half its nodes.
- The beam's own magnetic field (`self_magnetic_field=True`): the tracer deposits the
  beam's current as well, and its field, from Ampère's law of magnetostatics, acts on the
  next trace (`beamtracer.solver.self_field`): Ampère's integral in cylindrical and planar
  geometry, the vector potential on patchwork's `NodalPoisson` in 3D. The 3D kernel
  (Pyccel and CUDA) takes `has_j` and `current`. Checked against the generalised perveance
  `K (1 - f - beta^2)` of a drifting beam in all three geometries.

### Changed

- Field maps come from the takeout package (`takeout.FieldMap`, `takeout.field_map`), which
  holds the one copy of the layout, its validator and the interpolation that beamtracer and
  picnic share. `beamtracer.field_map` and `beamtracer.FieldMap` are removed (import
  `FieldMap` from takeout), and `FieldMapError` is `takeout.FormatError`. `FieldMapParams`
  and the `field_map` setting of `BeamOpticsConfig` are unchanged.

- beamtracer's kernels need cunumpy 0.8 (kernels that import mpiparticles' Pyccel blocks).

- The negative-ion beamlet example (`negative_ion_extraction_optics`) emits its beams an
  `emission_margin` (0.1 mm) inside the plasma grid's aperture instead of one cell inside
  it, so a finer mesh changes the discretisation, not the beam; the default beam is
  unchanged.

- Newton's method of the Boltzmann plasma species: where a chord step with the reused
  Jacobian stops contracting, it takes the Newton step by GMRES preconditioned by that
  Jacobian (to a relative residual of 1e-3) before refactorising. On the 3D extraction
  benchmark a fifth less time; the beam unchanged to 1e-7.

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

- `Layout.plot_cells(ax, scale=...)`: the cells of the grid and of every refined box, drawn
  by patchwork's `plot_patches` with `x` horizontal (also in cylindrical geometry);
  `Layout.hierarchy` is patchwork's hierarchy of the patches. The "Mesh refinement" tutorial
  draws its cells with it.

- Adaptive refinement (`adaptive_refinement=AdaptiveRefinementParams(...)`): every few
  Vlasov iterations the cells near the beam across which the field strength or the beam's
  charge density changes by more than a fraction of its largest value are tagged and
  covered with refined boxes (patchwork's Berger–Rigoutsos clustering); the field solver is
  rebuilt on them and the potential moved over (`beamtracer.solver.adaptive`). On the 3D
  extraction benchmark with 3600 trajectories: 0.3% from the mesh with half the cells
  everywhere in 19 s instead of 42 s. The "Mesh refinement" example shows the boxes it
  places.
- With `field_backend="multigrid"` the Newton–Krylov steps take their Jacobian through
  patchwork's `update_reaction` on refined boxes too (before, a reassembly and a full
  multigrid setup per step).

- The example "Mesh refinement": a beam extracted through an aperture on a coarse mesh with a
  refined box over the meniscus and the aperture, against uniform meshes (the divergence of
  the mesh with half the cells, with half its nodes).

- `field_backend="multigrid"`: patchwork's geometric multigrid with a compiled Gauss–Seidel
  V-cycle preconditions the Newton–Krylov steps, each with its own Jacobian (the lumped
  reaction changes only the diagonal, which the multigrid takes in about 10 ms). On the 3D
  extraction benchmark (55 000 nodes, 16 iterations) 7.0 s with `"amg"` → 2.6 s, IBSimu
  4.0 s on the same machine; the electrostatic quadrupole 0.31 s against IBSimu's 2.5 s. The
  beam is the same to 1e-6.

- Refined boxes on a 3D grid (`refinement` with `(x, y, z)` corners): patchwork's composite
  solve, a staggered field per box with ghost nodes from the coarser boxes, and the compiled
  3D tracer (and its CUDA version) steps and deposits on the finest box at a particle. A box
  over the whole domain gives the beam of the grid with half the cells to 1e-14.

- `field_backend="amg"`: patchwork's algebraic-multigrid backend (pyamg's classical AMG
  preconditioning CG or GMRES) instead of an LU factorisation, and with a plasma a
  Newton--Krylov method: every Newton step solves the current Jacobian by GMRES to 1e-3,
  preconditioned by the V-cycle of an AMG hierarchy kept across Newton steps, solves and
  Vlasov iterations and built again only when a step takes more than 12 iterations. The
  same potentials as the direct solve (to 2e-5 V on 100 V); on the 3D extraction benchmark
  (55 000 nodes) 8.5 s instead of 26 s, on the 3D solenoid 0.6 s instead of 3.1 s. Needs
  the `amg` extra (pyamg).

- `BeamOptics` on a 3D grid (`GridParams(ndim=3)`): electrodes as 3D shapes (`cuboid`,
  `sphere`, `cylinder`, `tube`, `cone`), the z faces in `FieldBCs`, beams emitted from a
  rectangle or a disc (`BeamSpecies3D`), the plasma species, solenoids and an axial profile
  on the x axis and magnetic field maps on (x, y, z) sampled onto the nodes,
  `symmetry_planes` (`"y_min"`, `"z_min"`), MPI over the trajectories, and the moments in
  either transverse plane (`beam_moments(x, axis="z")`). `BeamOptics.dataset()` returns the
  potential and the charge density as an xarray Dataset. A docs notebook runs an
  electrostatic quadrupole. Refinement, charge changes, an imposed electric field and
  `surface_fields` are 2D only for now.

- 3D ion optics, second part: `BeamTracer3D` and its `trace_trajectories_3d` kernel (Pyccel
  with OpenMP threads, a CUDA version; without Pyccel its source runs as Python). Every
  trajectory from start to end: RK4 or Boris (kick--rotate--kick) steps through the staggered
  electric field and a magnetic field on the nodes, the trilinear space-charge deposit at the
  Hermite midpoint, diagnostic planes, the exits and an electrode lookup in sub-cells of h/4
  (shared between MPI ranks), symmetry planes mirroring the field and the deposit.

- 3D ion optics, first part: `IonOpticsMesh3D` (electrodes as `inside(x, y, z)` tests, the
  wall distances bracketed along x, y and z, symmetry planes on the lower y and z faces),
  `IonOpticsPoisson` on it (patchwork's `NodalPoisson` in 3D: electrodes as embedded bodies,
  the Boltzmann plasma species by Newton's method with the reused Jacobian, the initial
  plasma region) and `StaggeredElectricField3D` (the edge fields along each axis,
  trilinear). An extruded planar case solves as in 2D (to 1e-10 V without electrodes), and
  the potential between concentric spheres converges to the exact one.

- `BeamOpticsConfig(symmetry_plane=True)`: a planar grid is half of a geometry mirrored at
  its `y` lower face (which needs a zero normal derivative). A trajectory crossing it
  continues at negative `y` in the mirrored field (E_y, B_x and B_z change sign) and
  deposits its charge mirrored, in the NumPy tracer and the compiled kernel (and its
  CUDA version), instead of leaving the mesh; the edge field is symmetric there, as on the
  axis. Before, a planar half-domain model (as IBSimu's mirrored boundaries make them)
  lost every trajectory focused across the plane.

- Mesh refinement: `BeamOpticsConfig(refinement=[RefinementParams(level, lower, upper)])`
  solves the field on boxes of finer cells (each level halves them) with patchwork's
  composite `NodalPoisson`, and every trajectory takes the field of, steps with the spacing
  of and deposits its charge on the finest box it is in (the compiled tracer, its CUDA
  version and the NumPy one). Each box's field reaches one coarse cell beyond it, so its
  edges see the coarser potential. On the negative-ion beamlet at 8 kV (best of 3, 4
  threads) a box of half the cells around the plasma grid gives the beam of the uniform
  0.05 mm mesh in 2.7 s instead of 9.5 s, and a second level in it that of the uniform
  0.025 mm mesh in 4.6 s instead of 38 s; without refinement the run is unchanged. The
  state keeps the potential and charge density of the base mesh, and the composite ones
  beside them.

- `BeamOpticsConfig(field_backend="petsc")`: under MPI the ranks share the field solve
  (patchwork's distributed backend: Krylov with algebraic multigrid, needs petsc4py) instead
  of each factorising the whole mesh; the Boltzmann Newton steps too. On IBSimu's `vlasov2d`
  case at 961 x 401 nodes the average field solve of a 5-iteration run goes from 0.49 s (LU on
  4 ranks, each factorising) to 0.16 s; a single later solve is still faster with the cached
  LU (0.03 s against 0.05 s), so `"lu"` stays the default. `vlasov_iterate` and
  `IonOpticsPoisson` take the backend and the communicator.

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

- `plane_moments`: the normalized emittance takes `beta gamma`, the mean of `gamma v_x / c`
  over the crossings, instead of `<v_x> / c`, which left out `gamma`; the mean energy is
  `(gamma - 1) m c^2` (½ m v² for slow particles).
- A Newton–Krylov step whose GMRES stops just short of its tolerance (SciPy checks the
  preconditioned residual) is taken when its true relative residual is below 0.1, instead of
  failing the solve: the 3D extraction benchmark at 420 000 nodes stopped there.
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
