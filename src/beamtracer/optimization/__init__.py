"""Design optimisation: searches over the parameters of a model, recorded and resumable.

The core knows nothing about the model it optimizes. A :class:`Design` of named
:class:`Parameter` builds a configuration with a user function; a :class:`Problem`
evaluates it with another user function, which returns an :class:`Evaluation`
(objectives, constraints ``<= 0``, a converged flag, the wall time and diagnostics), so
that a :class:`~beamtracer.BeamOptics` run, a PIC check or any other model can be the
evaluation. :func:`optimize` minimizes with SciPy's differential evolution and
:func:`scan` evaluates a grid, both in parallel processes, recording every evaluation in
a JSON-lines file that a later search resumes from (:class:`Record`,
:func:`load_record`). Failed and unconverged evaluations are penalised and never win.
Each evaluation of :func:`optimize` carries the generation that asked for it, and its
:class:`Result` has the path, the best design after each generation.
:func:`recheck` evaluates the few best designs of a search again with another
evaluation, e.g. a finer mesh, or a PIC model of the source from user code (beamtracer
does not import it): a function of the configuration that returns an
:class:`Evaluation`.

:mod:`beamtracer.optimization.einzel` is a first problem: an einzel lens solved with
:class:`~beamtracer.BeamOptics`; :mod:`beamtracer.optimization.beamlet` the second, the
extraction of a negative-ion beamlet from a plasma with the same searches.

A minimal search::

    from beamtracer.optimization import Design, Evaluation, Parameter, Problem, optimize

    def sphere(values):  # a module-level function, picklable for workers > 1
        return Evaluation(objectives={"objective": (values["x"] - 1.0) ** 2})

    design = Design((Parameter("x", -5.0, 5.0),))
    result = optimize(Problem(design, sphere), seed=1, record="sphere.jsonl")
    result.x  # {"x": 1.0...}
"""

from beamtracer.optimization.design import Design, Evaluation, Parameter, Problem
from beamtracer.optimization.record import Record, load_record, to_dataset
from beamtracer.optimization.search import Result, optimize, recheck, scan

__all__ = [
    "Design",
    "Evaluation",
    "Parameter",
    "Problem",
    "Record",
    "Result",
    "load_record",
    "optimize",
    "recheck",
    "scan",
    "to_dataset",
]
