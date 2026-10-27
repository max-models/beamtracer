"""The record of a search: every evaluation, one JSON line each, in the order evaluated.

A line holds the design's key (see :meth:`~beamtracer.optimization.Design.key`), the
parameters, objectives, constraints, flags, wall time and diagnostics of one
:class:`~beamtracer.optimization.Evaluation`. Python's :mod:`json` writes a non-finite
value as ``NaN`` or ``Infinity``, which it also reads. A search given an existing record
resumes from it: designs already recorded are taken from the record instead of being
evaluated again. :func:`load_record` reads a record as an :class:`xarray.Dataset` for
plots.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from beamtracer.optimization.design import Design, Evaluation


def _column(values: list[Any]) -> np.ndarray:
    """Return a column of numbers as floats (NaN for missing), or as objects."""
    if all(v is None or isinstance(v, bool | int | float) for v in values):
        if all(isinstance(v, bool) for v in values):
            return np.array(values, dtype=bool)
        return np.array([math.nan if v is None else float(v) for v in values])
    return np.array(["" if v is None else str(v) for v in values], dtype=object)


def to_dataset(evaluations: Iterable[Evaluation]) -> Any:
    """Return evaluations as an :class:`xarray.Dataset` along an ``evaluation`` dimension.

    The parameters, objectives, constraints and scalar diagnostics are variables under
    their own names (a name used twice gets its kind as a prefix, e.g.
    ``diagnostic_spot``), each with a ``kind`` attribute; ``converged``, ``ok``,
    ``feasible``, ``wall_time`` and ``failure`` (empty when none) are added. For a
    :class:`pandas.DataFrame` use the dataset's ``to_dataframe()``.
    """
    import xarray as xr

    evaluations = list(evaluations)
    groups = {
        "parameter": [e.parameters for e in evaluations],
        "objective": [e.objectives for e in evaluations],
        "constraint": [e.constraints for e in evaluations],
        "diagnostic": [
            {
                k: v
                for k, v in e.diagnostics.items()
                if v is None or isinstance(v, bool | int | float | str)
            }
            for e in evaluations
        ],
    }
    names: dict[str, list[str]] = {}
    for kind, rows in groups.items():
        names[kind] = list(dict.fromkeys(k for row in rows for k in row))
    taken: dict[str, int] = {}
    for kind_names in names.values():
        for name in kind_names:
            taken[name] = taken.get(name, 0) + 1
    data = {}
    for kind, rows in groups.items():
        for name in names[kind]:
            label = f"{kind}_{name}" if taken[name] > 1 else name
            column = _column([row.get(name) for row in rows])
            data[label] = ("evaluation", column, {"kind": kind})
    flags = {
        "converged": [e.converged for e in evaluations],
        "ok": [e.ok for e in evaluations],
        "feasible": [e.feasible for e in evaluations],
    }
    for name, values in flags.items():
        data[name] = ("evaluation", np.array(values, dtype=bool), {"kind": "flag"})
    data["wall_time"] = (
        "evaluation",
        np.array([e.wall_time for e in evaluations], dtype=float),
        {"kind": "flag", "units": "s"},
    )
    data["failure"] = (
        "evaluation",
        np.array([e.failure or "" for e in evaluations], dtype=object),
        {"kind": "flag"},
    )
    return xr.Dataset(data, coords={"evaluation": np.arange(len(evaluations))})


class Record:
    """The evaluations of a design, in memory and (with a ``path``) in a file.

    An existing file at ``path`` is read first; its parameter names must be those of the
    ``design``. New evaluations are appended to the file line by line as they arrive, so
    an interrupted search loses nothing it finished.
    """

    def _add(self, evaluation: Evaluation) -> str:
        key = self.design.key(evaluation.parameters)
        self._evaluations.append(evaluation)
        self._by_key.setdefault(key, evaluation)
        return key

    def __init__(self, design: Design, path: str | Path | None = None) -> None:
        """Read the evaluations already recorded at ``path``, if any."""
        self.design = design
        self.path = None if path is None else Path(path)
        self._evaluations: list[Evaluation] = []
        self._by_key: dict[str, Evaluation] = {}
        if self.path is not None and self.path.exists():
            for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                data = json.loads(line)
                names = tuple(data["parameters"])
                if set(names) != set(design.names):
                    raise ValueError(
                        f"{self.path}:{line_number} has the parameters {list(names)}, "
                        f"the design {list(design.names)}"
                    )
                self._add(Evaluation.from_dict(data))

    def get(self, key: str) -> Evaluation | None:
        """Return the recorded evaluation of a design key, or ``None``."""
        return self._by_key.get(key)

    def append(self, evaluations: Iterable[Evaluation]) -> None:
        """Record new evaluations (and append them to the file)."""
        lines = []
        for evaluation in evaluations:
            key = self._add(evaluation)
            lines.append(json.dumps({"key": key, **evaluation.to_dict()}))
        if self.path is not None and lines:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as file:
                file.write("\n".join(lines) + "\n")

    def to_dataset(self) -> Any:
        """Return the evaluations as an :class:`xarray.Dataset` (see :func:`to_dataset`)."""
        return to_dataset(self._evaluations)

    def __iter__(self) -> Iterator[Evaluation]:
        """Iterate over the evaluations in the order they were recorded."""
        return iter(self._evaluations)

    def __len__(self) -> int:
        """Return the number of recorded evaluations."""
        return len(self._evaluations)


def load_record(path: str | Path) -> Any:
    """Return the record at ``path`` as an :class:`xarray.Dataset` (see :func:`to_dataset`).

    The design is not needed: the evaluations are read as they were written.
    """
    evaluations = [
        Evaluation.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    return to_dataset(evaluations)
