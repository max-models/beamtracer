"""The shared sections of the configuration: the grid, the walls and the electrodes.

They read and write the configuration-file forms of picnic's sections of the same names
(``grid``, ``bcs_field``, ``objects``), so a picnic ``BeamOptics`` configuration file
is a beamtracer one. Values are SI; a ``{"value": ..., "unit": ...}`` entry of a
configuration file is converted with astropy.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Any, Literal, TypeAlias

import astropy.units as u
import numpy as np

ConfigDict: TypeAlias = dict[str, Any]

FaceType = Literal["dirichlet", "neumann"]

#: The 2D objects an electrode can be, by configuration-file ``type``.
OBJECT_TYPES = ("line", "rectangle", "triangle", "polygon", "circle", "aperture2d")


def to_si(value: Any, unit: str) -> Any:
    """Return ``value`` (a number or a nested list, tuple or dict of them) in SI units."""
    unit_obj = u.Unit(unit)

    def scalar(x: Any) -> float:
        si = u.Quantity(x, unit=unit_obj).si.value
        return int(si) if isinstance(x, int) and float(si).is_integer() else float(si)

    if isinstance(value, numbers.Real):
        return scalar(value)
    if isinstance(value, list):
        return [to_si(v, unit) for v in value]
    if isinstance(value, tuple):
        return tuple(to_si(v, unit) for v in value)
    if isinstance(value, dict):
        return {k: to_si(v, unit) for k, v in value.items()}
    raise TypeError(f"unsupported type {type(value)} for to_si")


def plain_value(value: Any) -> Any:
    """Return ``value`` with tuples as lists, for a configuration file."""
    if isinstance(value, (tuple, list)):
        return [plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: plain_value(item) for key, item in value.items()}
    return value


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _per_axis(value: Any) -> tuple:
    if isinstance(value, (int, float, bool)):
        return (value, value)
    values = tuple(np.asarray(value).ravel().tolist())
    if len(values) != 2:
        raise ValueError(f"need one value per axis (2), got {value!r}")
    return values


def _bound(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        value = to_si(value["values"], value["unit"])
    return tuple(float(v) for v in _per_axis(value))  # type: ignore[return-value]


@dataclass
class GridParams:
    """The 2D grid of nodes: nodes per axis and extent in metres.

    ``coordinates`` is ``"cartesian"`` (planar, per unit depth) or ``"cylindrical"``
    (``(x, r)`` with the axis at ``r = 0``). A scalar ``num_gridpoints`` or bound
    applies to both axes; bounds may also be ``{"values": ..., "unit": "mm"}``.
    ``ndim`` is always 2 (kept so that picnic's grid entries read unchanged).
    """

    ndim: int = 2
    num_gridpoints: Any = 64
    lower_bound: Any = 0.0
    upper_bound: Any = 32.0e-3
    coordinates: str = "cartesian"

    def __post_init__(self) -> None:
        """Normalize every per-axis value to a tuple and check them."""
        if int(self.ndim) != 2:
            raise ValueError(f"the beam tracer needs a 2D grid, got ndim={self.ndim}")
        self.ndim = 2
        self.num_gridpoints = tuple(int(n) for n in _per_axis(self.num_gridpoints))
        self.lower_bound = _bound(self.lower_bound)
        self.upper_bound = _bound(self.upper_bound)
        if any(
            hi <= lo for lo, hi in zip(self.lower_bound, self.upper_bound, strict=True)
        ):
            raise ValueError(
                f"upper_bound {self.upper_bound} must exceed lower_bound "
                f"{self.lower_bound} on every axis"
            )

    @property
    def deltas(self) -> list[float]:
        """Return the node spacing along each axis, ``(upper - lower) / (n - 1)``."""
        return [
            (b1 - b0) / (n - 1)
            for b0, b1, n in zip(
                self.lower_bound, self.upper_bound, self.num_gridpoints, strict=True
            )
        ]

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form (bounds in metres)."""
        return {
            "ndim": self.ndim,
            "num_gridpoints": list(self.num_gridpoints),
            "lower_bound": {"values": list(self.lower_bound), "unit": "m"},
            "upper_bound": {"values": list(self.upper_bound), "unit": "m"},
            "coordinates": self.coordinates,
        }

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> GridParams:
        """Create the grid from its configuration-file entry.

        picnic's entries ``nvel`` and ``domain_decomposition`` are accepted and ignored.
        """
        keys = ("ndim", "num_gridpoints", "lower_bound", "upper_bound", "coordinates")
        unknown = set(params) - set(keys) - {"nvel", "domain_decomposition"}
        if unknown:
            raise KeyError(f"unknown grid entries {sorted(unknown)}")
        return cls(**{key: params[key] for key in keys if key in params})


class SymbolicSpaceFunction:
    """A number or a symbolic expression of ``x``, ``y`` (and ``z``, ``t``) in SI units.

    ``unit`` scales the value (``"kV"``: the expression is in kilovolts); ``variables``
    gives the units of the coordinates in the expression, e.g. ``{"y": {"unit": "mm"}}``.
    Expressions are parsed with sympy.
    """

    def __init__(
        self,
        expression: float | str,
        unit: str | None = None,
        variables: Mapping[str, Any] | None = None,
    ) -> None:
        """Parse the expression and its units."""
        self._expression = expression
        self._unit = unit
        self._variables = None if variables is None else dict(variables)
        self._factor = 1.0 if not unit else float(to_si(1.0, unit))
        self._si_factors = {"x": 1.0, "y": 1.0, "z": 1.0, "t": 1.0}
        for name, variable in (self._variables or {}).items():
            if name not in self._si_factors:
                raise ValueError(f"unknown variable {name!r} (x, y, z, t)")
            self._si_factors[name] = 1.0 / float(to_si(1.0, variable["unit"]))
        if self._variables is None and _is_number(expression):
            value = float(expression)
            # a constant, like sympy's lambdify of a number
            self._f = lambda x, y, z, t: value
        else:
            import sympy as sp

            symbols = sp.symbols("x y z t")
            self._f = sp.lambdify(symbols, sp.sympify(expression), modules=["numpy"])

    def evaluate(
        self, x: Any = None, y: Any = None, z: Any = None, t: Any = None
    ) -> Any:
        """Evaluate the value at the points ``(x, y, z)`` and time ``t`` [SI]."""
        factors = self._si_factors

        def scaled(value: Any, name: str) -> Any:
            return None if value is None else value * factors[name]

        return (
            self._f(
                scaled(x, "x"),
                scaled(y, "y"),
                scaled(z, "z"),
                0.0 if t is None else t * factors["t"],
            )
            * self._factor
        )

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form (``value`` or ``func``)."""
        if self._variables is None and _is_number(self._expression):
            return {"value": float(self._expression), "unit": self._unit or ""}
        func: ConfigDict = {"expression": str(self._expression), "unit": self._unit}
        if self._variables is not None:
            func["variables"] = self._variables
        return {"func": func}

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> SymbolicSpaceFunction:
        """Create the value from ``{"value", "unit"}`` or ``{"func": {...}}``."""
        if "value" in params:
            return cls(params["value"], params.get("unit"))
        if "func" in params:
            func = params["func"]
            return cls(func["expression"], func.get("unit"), func.get("variables"))
        raise ValueError("a field boundary needs a value or a func")


class FieldBoundaryCondition:
    """A Dirichlet (potential) or Neumann (normal derivative) condition on a face."""

    def __init__(self, bc_type: FaceType, value: SymbolicSpaceFunction) -> None:
        """Store the type and the value."""
        if bc_type not in ("dirichlet", "neumann"):
            raise ValueError(
                f"the beam tracer's faces are dirichlet or neumann, got {bc_type!r}"
            )
        self._bc_type: FaceType = bc_type
        self._value = value

    @staticmethod
    def dirichlet(
        value: float | str = 0.0, unit: str = "V", variables: Mapping | None = None
    ) -> FieldBoundaryCondition:
        """Return a Dirichlet condition: a potential or an expression of x, y."""
        return FieldBoundaryCondition(
            "dirichlet", SymbolicSpaceFunction(value, unit, variables)
        )

    @staticmethod
    def neumann(
        normal_derivative: float | str = 0.0,
        unit: str = "V/m",
        variables: Mapping | None = None,
    ) -> FieldBoundaryCondition:
        """Return a Neumann condition on the normal derivative of the potential."""
        return FieldBoundaryCondition(
            "neumann", SymbolicSpaceFunction(normal_derivative, unit, variables)
        )

    @property
    def bc_type(self) -> FaceType:
        """Return ``"dirichlet"`` or ``"neumann"``."""
        return self._bc_type

    @property
    def value(self) -> SymbolicSpaceFunction:
        """Return the value of the condition."""
        return self._value

    def evaluate(
        self, x: Any = None, y: Any = None, z: Any = None, t: Any = None
    ) -> Any:
        """Evaluate the value at the points ``(x, y)``."""
        return self._value.evaluate(x=x, y=y, z=z, t=t)


class FieldBCs:
    """The conditions on the four faces of the grid."""

    def __init__(
        self,
        x_lower: FieldBoundaryCondition,
        x_upper: FieldBoundaryCondition,
        y_lower: FieldBoundaryCondition,
        y_upper: FieldBoundaryCondition,
    ) -> None:
        """Store the four faces."""
        self._bc = {
            "x": {"min": x_lower, "max": x_upper},
            "y": {"min": y_lower, "max": y_upper},
        }

    @staticmethod
    def grounded() -> FieldBCs:
        """Return 0 V on every face."""
        zero = FieldBoundaryCondition.dirichlet(0.0)
        return FieldBCs(zero, zero, zero, zero)

    def get_bc(self, dir: str, side: str) -> FieldBoundaryCondition:
        """Return the condition of one side (``"min"``, ``"max"``) of one axis."""
        return self._bc[dir][side]

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form, the inverse of :meth:`from_dict`."""
        params: ConfigDict = {}
        for axis, sides in self._bc.items():
            lower = sides["min"]
            entry: ConfigDict = {"type": lower.bc_type}
            for key, bc in sides.items():
                side = bc.value.to_dict()
                if bc.bc_type != lower.bc_type:
                    side["type"] = bc.bc_type
                entry[key] = side
            params[axis] = entry
        return params

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> FieldBCs:
        """Create the faces from ``{axis: {"type", "min": {...}, "max": {...}}}``."""
        faces: dict[str, FieldBoundaryCondition] = {}
        for axis in ("x", "y"):
            if axis not in params:
                raise ValueError(f"field boundaries need the {axis!r} axis")
            item = params[axis]
            axis_type = item.get("type")
            for key, name in (("min", "lower"), ("max", "upper")):
                side = item[key]
                bc_type = side.get("type", axis_type)
                if bc_type is None:
                    raise ValueError(f"field boundary '{axis}/{key}' needs a type")
                faces[f"{axis}_{name}"] = FieldBoundaryCondition(
                    bc_type, SymbolicSpaceFunction.from_dict(side)
                )
        return cls(**faces)


def _plain(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    return [_plain(item) for item in value]


#: The geometry entries each shape type takes (simgeometry's constructor arguments).
_OBJECT_ARGUMENTS = {
    "line": ("midpoint", "side_lengths"),
    "rectangle": ("midpoint", "side_lengths"),
    "triangle": ("vertices",),
    "polygon": ("vertices",),
    "circle": ("midpoint", "radius"),
    "aperture2d": ("axis", "position", "vertices"),
}

#: The geometry entries of a shape, in the order of :class:`ShapeParams`.
_GEOMETRY = ("vertices", "midpoint", "side_lengths", "radius", "axis", "position")


@dataclass
class ShapeParams:
    """One simgeometry shape of an electrode, without a potential.

    ``type`` is one of :data:`OBJECT_TYPES`; the geometry entries are those of the
    simgeometry class (lengths in metres). An electrode of several shapes,
    :meth:`ObjectParams.union`, lists them.
    """

    type: str
    vertices: Any = None
    midpoint: Any = None
    side_lengths: Any = None
    radius: float | None = None
    axis: str | None = None
    position: float | None = None

    def __post_init__(self) -> None:
        """Check the type."""
        if self.type not in OBJECT_TYPES:
            raise ValueError(
                f"shape type must be one of {OBJECT_TYPES}, got {self.type!r}"
            )

    def arguments(self) -> dict[str, Any]:
        """Return the simgeometry constructor arguments of the shape."""
        return {name: getattr(self, name) for name in _OBJECT_ARGUMENTS[self.type]}

    def build(self, bc: Any = None) -> Any:
        """Return the simgeometry object (with ``contains_points``)."""
        from simgeometry.objects import (
            Aperture2D,
            Circle,
            Line,
            Polygon,
            Rectangle,
            Triangle,
        )

        classes = {
            "line": Line,
            "rectangle": Rectangle,
            "triangle": Triangle,
            "polygon": Polygon,
            "circle": Circle,
            "aperture2d": Aperture2D,
        }
        return classes[self.type](**self.arguments(), bc=bc)

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form (lengths in metres)."""
        params: ConfigDict = {"type": self.type}
        if self.axis is not None:
            params["axis"] = self.axis
        for key in ("vertices", "midpoint", "side_lengths"):
            value = getattr(self, key)
            if value is not None:
                params[key] = {"values": _plain(value), "unit": "m"}
        for key in ("radius", "position"):
            value = getattr(self, key)
            if value is not None:
                params[key] = {"value": float(value), "unit": "m"}
        return params

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> ShapeParams:
        """Create the shape from its configuration-file entry (``bc`` is ignored)."""

        def entry(name: str) -> Any:
            if name not in params:
                return None
            item = params[name]
            value = item["values"] if "values" in item else item["value"]
            return to_si(value, item["unit"])

        return cls(
            type=params["type"],
            vertices=entry("vertices"),
            midpoint=entry("midpoint"),
            side_lengths=entry("side_lengths"),
            radius=entry("radius"),
            axis=params.get("axis"),
            position=entry("position"),
        )


@dataclass
class ObjectParams:
    """An electrode: a simgeometry object held at a potential (``bc_type="dirichlet"``).

    ``type`` is one of :data:`OBJECT_TYPES`; the geometry entries are those of the
    simgeometry class (lengths in metres). ``type="union"`` is an electrode of several
    shapes at one potential, the :class:`ShapeParams` (or their configuration-file
    entries) in ``shapes``, e.g. a tube with rounded ends from a rectangle and two
    circles; see :meth:`union`. It is one electrode for the field, the hits and the
    diagnostics.
    """

    type: str
    bc_type: str
    bc_value: float | None = None
    vertices: Any = None
    midpoint: Any = None
    side_lengths: Any = None
    radius: float | None = None
    axis: str | None = None
    position: float | None = None
    shapes: list[ShapeParams] | None = None

    def __post_init__(self) -> None:
        """Check the type and read the shapes of a union."""
        if self.type == "union":
            if not self.shapes:
                raise ValueError("a union needs at least one shape")
            given = {name for name in _GEOMETRY if getattr(self, name) is not None}
            if given:
                raise ValueError(
                    f"the geometry of a union is in its shapes, got {sorted(given)}"
                )
            self.shapes = [
                shape
                if isinstance(shape, ShapeParams)
                else ShapeParams.from_dict(shape)
                for shape in self.shapes
            ]
            return
        if self.type not in OBJECT_TYPES:
            raise ValueError(
                f"object type must be one of {OBJECT_TYPES} or 'union', "
                f"got {self.type!r}"
            )
        if self.shapes is not None:
            raise ValueError(f"only a union has shapes, not a {self.type!r}")

    @classmethod
    def union(
        cls,
        shapes: list[ShapeParams],
        bc_value: float,
        bc_type: str = "dirichlet",
    ) -> ObjectParams:
        """Return one electrode made of ``shapes``, all at ``bc_value``."""
        return cls(type="union", bc_type=bc_type, bc_value=bc_value, shapes=shapes)

    @property
    def shape_list(self) -> list[ShapeParams]:
        """Return the shapes of the electrode: those of a union, else its own."""
        if self.shapes is not None:
            return list(self.shapes)
        geometry = {name: getattr(self, name) for name in _GEOMETRY}
        return [ShapeParams(self.type, **geometry)]

    def build(self) -> Any:
        """Return the simgeometry object (with ``contains_points``).

        A union is simgeometry's Boolean union (``|``) of its shapes.
        """
        objects = [shape.build(bc=self.bc_value) for shape in self.shape_list]
        return reduce(operator.or_, objects)

    def get_filtered_obj_params(self) -> dict[str, Any]:
        """Return the simgeometry constructor arguments of the object, with ``bc``.

        A union has none: build it with :meth:`build`.
        """
        if self.type == "union":
            raise ValueError("a union has no constructor arguments; use build()")
        arguments = self.shape_list[0].arguments()
        arguments["bc"] = self.bc_value
        return arguments

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form (lengths in metres)."""
        if self.shapes is not None:
            params: ConfigDict = {
                "type": self.type,
                "shapes": [shape.to_dict() for shape in self.shapes],
            }
        else:
            params = self.shape_list[0].to_dict()
        if self.bc_type == "neumann":
            bc = {"normal_derivative": {"value": self.bc_value, "unit": "V/m"}}
        else:
            bc = {"potential": {"value": self.bc_value, "unit": "V"}}
        params["bc"] = {"type": self.bc_type, **bc}
        return params

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> ObjectParams:
        """Create the object from its configuration-file entry.

        A ``"union"`` lists the entries of its shapes (without ``bc``) in ``shapes``.
        """
        bc_type = params["bc"]["type"]
        key = "normal_derivative" if bc_type == "neumann" else "potential"
        if key not in params["bc"]:
            raise ValueError(f"an object with a {bc_type!r} condition needs bc.{key}")
        value = params["bc"][key]
        bc_value = to_si(value["value"], value["unit"])
        if params["type"] == "union":
            return cls.union(
                [ShapeParams.from_dict(shape) for shape in params["shapes"]],
                bc_value,
                bc_type,
            )
        shape = ShapeParams.from_dict(params)
        geometry = {name: getattr(shape, name) for name in _GEOMETRY}
        return cls(type=shape.type, bc_type=bc_type, bc_value=bc_value, **geometry)
