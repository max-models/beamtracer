"""Evaluating designs in worker processes, robust to crashes, hangs and a dying parent.

:class:`WorkerPool` wraps a :class:`~concurrent.futures.ProcessPoolExecutor`:

- its workers exit when the main process is gone (a search killed from outside leaves no
  orphans waiting for work);
- a worker that dies (a segmentation fault, the out-of-memory killer) breaks the pool;
  the pool is restarted and the batch evaluated again one design at a time, so that only
  the designs that kill their worker become failed evaluations;
- with a ``timeout``, an evaluation that runs longer is interrupted and becomes a failed
  evaluation (a timer signal, so only on Unix, in the worker's main thread; code that
  does not return to Python, e.g. a long sparse factorization, is interrupted when it
  does);
- an interrupted search (``KeyboardInterrupt``) terminates the workers.
"""

from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from beamtracer.optimization.design import Evaluation, Problem


class EvaluationTimeout(Exception):
    """An evaluation ran longer than its timeout."""


def _exit_with(parent: int) -> None:
    """Start a thread that ends this worker when the process ``parent`` is gone."""
    if sys.platform == "win32":  # os.kill(pid, 0) would terminate the parent there
        return

    def watch() -> None:
        while True:
            try:
                os.kill(parent, 0)
            except ProcessLookupError:
                os._exit(1)
            except PermissionError:  # the pid was reused by another user's process
                os._exit(1)
            time.sleep(1.0)

    threading.Thread(target=watch, daemon=True, name="exit-with-parent").start()


def failed(problem: Problem, values: dict[str, float], message: str) -> Evaluation:
    """Return the failed evaluation of a design: NaN objectives and constraints."""
    return Evaluation(
        objectives={name: math.nan for name in problem.objectives},
        constraints={name: math.nan for name in problem.constraints},
        converged=False,
        failure=message,
        parameters=values,
    )


def evaluate(
    problem: Problem, values: dict[str, float], timeout: float | None
) -> Evaluation:
    """Return ``problem.evaluate(values)``, a failed evaluation after ``timeout`` [s].

    The timeout needs a timer signal: it is ignored off the main thread and on Windows.
    """
    timed = (
        timeout is not None
        and math.isfinite(timeout)
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not timed:
        return problem.evaluate(values)

    def interrupt(signum: int, frame: Any) -> None:
        raise EvaluationTimeout(f"the evaluation took longer than {timeout} s")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))  # type: ignore[arg-type]
    try:
        return problem.evaluate(values)
    except EvaluationTimeout as error:  # fired outside the evaluation's own handler
        return failed(problem, values, f"EvaluationTimeout: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


class WorkerPool:
    """Evaluate designs of a problem serially (``workers=1``) or in worker processes.

    ``workers`` of ``-1`` is one per CPU; ``mp_context`` is the :mod:`multiprocessing`
    context that starts them; ``timeout`` [s] bounds every evaluation.
    """

    def __init__(
        self,
        problem: Problem,
        workers: int = 1,
        mp_context: Any = None,
        timeout: float | None = None,
    ) -> None:
        """Start the workers (none for ``workers=1``)."""
        if workers == -1:
            workers = os.cpu_count() or 1
        if workers < 1:
            raise ValueError(f"workers must be positive or -1, got {workers}")
        self.problem = problem
        self.workers = workers
        self.mp_context = mp_context
        self.timeout = timeout
        self._executor: ProcessPoolExecutor | None = None
        if workers > 1:
            self._start()

    def _start(self) -> None:
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=self.mp_context,
            initializer=_exit_with,
            initargs=(os.getpid(),),
        )

    def _restart(self) -> None:
        self.close(terminate=True)
        self._start()

    def map(self, values: Sequence[dict[str, float]]) -> list[Evaluation]:
        """Return the evaluations of the designs ``values``, in order."""
        if self._executor is None:
            return [evaluate(self.problem, v, self.timeout) for v in values]
        jobs = [(self.problem, v, self.timeout) for v in values]
        try:
            return list(self._executor.map(evaluate, *zip(*jobs, strict=True)))
        except BrokenProcessPool:
            self._restart()
        except BaseException:
            self.close(terminate=True)
            raise
        # a worker died: evaluate one design at a time to find the ones that kill it
        results = []
        for job in jobs:
            assert self._executor is not None
            try:
                results.append(self._executor.submit(evaluate, *job).result())
            except BrokenProcessPool as error:
                self._restart()
                message = f"BrokenProcessPool: the worker process died ({error})"
                results.append(failed(self.problem, job[1], message))
        return results

    def close(self, terminate: bool = False) -> None:
        """Shut the workers down; with ``terminate``, without waiting for them."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        if not terminate:
            executor.shutdown(wait=True, cancel_futures=True)
            return
        processes = list(getattr(executor, "_processes", {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> WorkerPool:
        """Return the pool."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Shut the workers down (terminate them after an exception)."""
        self.close(terminate=exc[0] is not None)
