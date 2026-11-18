# IBSimu benchmarks

Cases from IBSimu's Python examples, set up in beamtracer and in IBSimu on the same mesh, to
check that beamtracer reproduces IBSimu and to time both. IBSimu runs through its Python
bindings (the field solve and the tracing run in its C++ library), with trajectory saving
off, as beamtracer does.

| Case | IBSimu example | Geometry | What it checks |
|---|---|---|---|
| `vlasov2d` | `vlasov2d.py` | planar, mirrored | extraction with space charge, three electrodes |
| `lens_2d` | `sim_2d_lens.py` | planar | an einzel lens |
| `lens_cyl` | `sim_cyl_lens.py` | cylindrical | an einzel lens in a pipe, space charge |
| `two_stage` | `sim_two_stage_accel.py` | planar | acceleration from 10 eV to 50 keV in two gaps |
| `iter_nbi` | `sim_iter_nbi.py` | cylindrical | seven chamfered electrodes to 120 keV |
| `plasma2d` | `plasma2d.py` | planar, mirrored | positive-ion extraction, Boltzmann electrons |
| `plasmacyl` | `plasmacyl.py` | cylindrical | He+ extraction, Boltzmann electrons |
| `solenoid` | `sim_solenoid.py` | cylindrical | an axial magnetic field |
| `sector_magnet` | `sim_sector_magnet.py` | planar | a magnetic field map |

IBSimu has 3D cases too (`sim_3d_quad.py`, `viz_solenoid_3d.py`); beamtracer's ion optics
are 2D (planar and cylindrical), so they follow with a 3D `BeamOptics`.

## What is compared

Without saved trajectories IBSimu keeps only where every particle ended, so both codes are
compared at the exit face `x = x_max` of the mesh, on the current-weighted moments of the
particles crossing it: the current, the mean and rms transverse position (`y` or `r`), the
mean and rms angle, the rms emittance and the mean energy. The moments of a beam symmetric about
`y = 0` are taken about 0 (a planar trajectory may continue across that plane in one code and
be mirrored in the other). The potentials are compared at the nodes at least a cell away
from every electrode, relative to the potential's span: inside and on an electrode the codes
differ by convention, not by their solutions.

Each case module in `cases/` describes its case once: the electrodes as polygons, the
faces, the beam, a plasma or a magnetic field. `common.run_ibsimu` and
`common.run_beamtracer` build it in each code, IBSimu's inside tests from the same polygons.
Every electrode is shrunk by 0.37 of a cell, in both codes, so that no surface passes
through a node: there the codes would classify the node by convention (or by rounding)
rather than by the geometry, and in the examples many surfaces lie on nodes. Where an
example traces only a few hundred trajectories, the case traces more, so that the moments
are not dominated by sampling noise. Each case lists its tolerances and where it departs
from the example (`notes`).

What the cases found:

- beamtracer has no symmetry plane in planar geometry: a trajectory crossing `y = 0`
  leaves the mesh, where IBSimu mirrors it. `plasma2d`, whose meniscus focuses the beam
  across the plane, fails for that reason.
- In `iter_nbi` the outermost trajectories graze a corner of the last electrode; IBSimu,
  testing for collisions only at its step points, lets them through. Its Gauss-Seidel
  solver also stops about 35 V short on that long mesh, so the case uses BiCGSTAB.
- Setups with a magnetic field (`solenoid`, `sector_magnet`) still use beamtracer's NumPy
  tracer and are slower than IBSimu.

## Running

IBSimu's Python bindings come from max-models/ibsimu (see its `PYTHON.md`): build the
library and install the package into the environment, or put the checkout on `PYTHONPATH`
and its `src/.libs` on `LD_LIBRARY_PATH`. Then, from the beamtracer checkout:

```bash
python benchmarks/run.py                        # every case, both codes
python benchmarks/run.py lens_cyl plasmacyl     # some cases
python benchmarks/run.py vlasov2d --scale 4 --repeats 3 --threads 4 --integrator boris
python benchmarks/run.py --codes beamtracer     # beamtracer alone, no comparison
```

`--scale n` divides the cells by `n`, `--particles` overrides the trajectories, `--threads`
sets the threads of both codes (`--ibsimu-threads` IBSimu's alone). Each code runs each case
in its own process under a scope-profiler session; the profiles (`<case>_<code>.h5`), the
potentials and exit beams (`.npz`) and the report (`results.md`, `results.json`) go to
`--out` (`benchmark_results/`). The exit status is 1 when a case does not agree. The times
are the best of `--repeats` runs; take at least 3 for timings, since the first run of a
process includes one-off costs (loading the compiled kernels, the first factorisation).
