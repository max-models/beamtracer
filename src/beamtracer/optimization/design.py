"""Designs, evaluations and problems: what the search varies, measures and minimizes.

A :class:`Design` is a set of named :class:`Parameter` with bounds; it turns parameter
values into a model configuration with a user-supplied ``build`` function. A
:class:`Problem` pairs a design with an evaluation function, which takes that
configuration and returns an :class:`Evaluation`: objectives, constraints, a converged
flag and diagnostics. :meth:`Problem.evaluate` never raises: an exception becomes a
failed evaluation, which the searches rank behind every successful one.

Nothing here knows about a model: any picklable function of a configuration can be the
evaluation, a :class:`~beamtracer.BeamOptics` run or a code beamtracer does not import.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import numpy as np

#: Parameter values: ``{name: value}``, or a sequence or array in the design's order.
Values: TypeAlias = Mapping[str, float] | Sequence[float] | np.ndarray

#: The digits of the normalized parameters that identify a design in a record.
KEY_DECIMALS = 10


@dataclass(frozen=True)
class Parameter:
    """A design parameter ``name`` between ``lower`` and ``upper`` (in ``unit``)."""

    name: str
    lower: float
    upper: float
    unit: str = ""

    def __post_init__(self) -> None:
        """Check the bounds."""
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError(f"the bounds of {self.name!r} must be finite")
        if not self.lower < self.upper:
            raise ValueError(
                f"{self.name!r} needs lower < upper, got {self.lower} and {self.upper}"
            )


@dataclass(frozen=True)
class Design:
    """The parameters of a design, and the function that builds its configuration.

    ``build(values)`` receives the parameter values as a ``{name: value}`` dictionary and
    returns whatever the evaluation function takes, e.g. a
    :class:`~beamtracer.BeamOpticsConfig`; without it the evaluation receives the values.
    For parallel searches ``build`` must be picklable: a module-level function, a
    :func:`functools.partial` of one, or a method of a picklable (e.g. frozen dataclass)
    object, not a lambda or a function defined in a notebook.
    """

    parameters: tuple[Parameter, ...]
    build: Callable[[dict[str, float]], Any] | None = None

    def __post_init__(self) -> None:
        """Store the parameters as a tuple and check their names."""
        parameters = tuple(self.parameters)
        object.__setattr__(self, "parameters", parameters)
        if not parameters:
            raise ValueError("a design needs at least one parameter")
        names = [p.name for p in parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"parameter names must be unique, got {names}")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the parameter names, in order."""
        return tuple(p.name for p in self.parameters)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Return the ``(lower, upper)`` bounds, in order."""
        return [(p.lower, p.upper) for p in self.parameters]

    def vector(self, values: Values) -> np.ndarray:
        """Return the values (a mapping by name, or a sequence in order) as an array."""
        if isinstance(values, Mapping):
            missing = set(self.names) - set(values)
            if missing:
                raise KeyError(f"missing parameters {sorted(missing)}")
            x = np.array([float(values[name]) for name in self.names])
        else:
            x = np.asarray(values, dtype=float).ravel()
            if x.size != len(self):
                raise ValueError(f"need {len(self)} values, got {x.size}")
        return x

    def values(self, x: Values) -> dict[str, float]:
        """Return the values as ``{name: value}``."""
        return dict(zip(self.names, map(float, self.vector(x)), strict=True))

    def normalized(self, x: Values) -> np.ndarray:
        """Return the values mapped to ``[0, 1]`` between the bounds."""
        lower, upper = np.array(self.bounds).T
        return (self.vector(x) - lower) / (upper - lower)

    def key(self, x: Values) -> str:
        """Return the identity of a design in a record.

        A hash of the parameter names and of the normalized values rounded to
        :data:`KEY_DECIMALS` digits: two designs closer than about ``1e-10`` of the
        parameter ranges are the same.
        """
        rounded = np.round(self.normalized(x), KEY_DECIMALS) + 0.0  # no "-0.0"
        text = json.dumps([self.names, [f"{v:.{KEY_DECIMALS}f}" for v in rounded]])
        return hashlib.sha1(text.encode()).hexdigest()[:16]

    def configuration(self, values: Values) -> Any:
        """Return the configuration of the values (the values without ``build``)."""
        named = self.values(values)
        return named if self.build is None else self.build(named)

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self.parameters)


def _plain(value: Any) -> Any:
    """Return ``value`` with NumPy scalars and arrays as Python numbers and lists."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class Evaluation:
    """The outcome of evaluating one design.

    ``objectives`` (minimized) and ``constraints`` are ``{name: value}``; a constraint is
    satisfied when its value is ``<= 0`` (write ``g(x) <= limit`` as
    ``g(x) / limit - 1``, so that the values are of order one). ``converged`` is the
    model's own verdict, ``failure`` the message of an exception, ``wall_time`` [s] the
    time of the evaluation and ``diagnostics`` anything else worth keeping (numbers,
    strings or flags, recorded with the evaluation). ``parameters`` are filled in by
    :meth:`Problem.evaluate`.
    """

    objectives: Mapping[str, float]
    constraints: Mapping[str, float] = field(default_factory=dict)
    converged: bool = True
    wall_time: float = float("nan")
    failure: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store the values as floats."""
        object.__setattr__(
            self, "objectives", {k: float(v) for k, v in self.objectives.items()}
        )
        object.__setattr__(
            self, "constraints", {k: float(v) for k, v in self.constraints.items()}
        )
        object.__setattr__(
            self, "parameters", {k: float(v) for k, v in self.parameters.items()}
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "converged", bool(self.converged))

    @property
    def ok(self) -> bool:
        """Whether the evaluation converged, raised nothing and has finite objectives."""
        return (
            self.converged
            and self.failure is None
            and all(math.isfinite(v) for v in self.objectives.values())
        )

    @property
    def violation(self) -> float:
        """Return the sum of the positive constraint values (``inf`` for a NaN one)."""
        return float(
            sum(
                math.inf if math.isnan(v) else max(v, 0.0)
                for v in self.constraints.values()
            )
        )

    @property
    def feasible(self) -> bool:
        """Whether the evaluation is :attr:`ok` and satisfies every constraint."""
        return self.ok and self.violation == 0.0

    def rank(self, objective: str) -> tuple[int, float]:
        """Return the sort key of the searches: lower is better.

        Feasible evaluations by ``objective``, then infeasible ones by their constraint
        violation, then failed or unconverged ones: a penalised evaluation never ranks
        before a feasible one.
        """
        if self.feasible:
            return (0, self.objectives[objective])
        if self.ok:
            return (1, self.violation)
        return (2, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as a JSON-friendly dictionary."""
        return {
            "parameters": dict(self.parameters),
            "objectives": dict(self.objectives),
            "constraints": dict(self.constraints),
            "converged": self.converged,
            "feasible": self.feasible,
            "wall_time": self.wall_time,
            "failure": self.failure,
            "diagnostics": _plain(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Evaluation:
        """Return the evaluation of :meth:`to_dict`."""
        return cls(
            objectives=data["objectives"],
            constraints=data.get("constraints", {}),
            converged=data.get("converged", True),
            wall_time=data.get("wall_time", float("nan")),
            failure=data.get("failure"),
            diagnostics=data.get("diagnostics", {}),
            parameters=data.get("parameters", {}),
        )


@dataclass(frozen=True)
class Problem:
    """A design, the function that evaluates it, and what to minimize.

    ``function(configuration)`` returns an :class:`Evaluation` for the configuration
    that ``design.build`` made. ``objectives`` names the objectives, the first being the
    one the searches minimize (several are kept for multi-objective searches later);
    ``constraints`` names the constraints the searches enforce, which every successful
    evaluation must return; the other constraints an evaluation returns are kept as
    diagnostics. For parallel searches the function must be picklable, like
    ``design.build``.
    """

    design: Design
    function: Callable[[Any], Evaluation]
    objectives: tuple[str, ...] | str = ("objective",)
    constraints: tuple[str, ...] | str = ()

    def __post_init__(self) -> None:
        """Store the names as tuples."""
        for name in ("objectives", "constraints"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, (value,) if isinstance(value, str) else tuple(value)
            )
        if not self.objectives:
            raise ValueError("a problem needs at least one objective")

    @property
    def objective(self) -> str:
        """Return the name of the objective the searches minimize."""
        return self.objectives[0]

    def evaluate(self, values: Values) -> Evaluation:
        """Build and evaluate one design; never raise.

        An exception of the build or of the function, a result that is not an
        :class:`Evaluation`, or a missing objective or constraint becomes a failed
        evaluation (``failure`` holds the message, the objectives are NaN).
        """
        named = self.design.values(values)
        start = time.perf_counter()
        try:
            result = self.function(self.design.configuration(named))
            if not isinstance(result, Evaluation):
                raise TypeError(
                    f"the evaluation function returned {type(result).__name__}, "
                    "not an Evaluation"
                )
            missing = (set(self.objectives) - set(result.objectives)) | (
                set(self.constraints) - set(result.constraints)
            )
            if missing:
                raise KeyError(f"the evaluation is missing {sorted(missing)}")
            # the constraints this problem does not enforce are diagnostics
            extra = {
                k: v for k, v in result.constraints.items() if k not in self.constraints
            }
            result = replace(
                result,
                constraints={k: result.constraints[k] for k in self.constraints},
                diagnostics={**extra, **result.diagnostics},
            )
        except Exception as error:  # noqa: BLE001 - any failure is a penalised point
            result = Evaluation(
                objectives={name: math.nan for name in self.objectives},
                constraints={name: math.nan for name in self.constraints},
                converged=False,
                failure=f"{type(error).__name__}: {error}",
                diagnostics={"traceback": traceback.format_exc(limit=-3)},
            )
        return replace(result, parameters=named, wall_time=time.perf_counter() - start)
