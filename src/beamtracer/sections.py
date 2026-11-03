"""The shared sections of the configuration: the grid, the walls and the electrodes.

They read and write the configuration-file forms of picnic's sections of the same names
(``grid``, ``bcs_field``, ``objects``), so a picnic ``BeamOptics`` configuration file
is a beamtracer one. Values are SI: lengths in m, potentials in V. Every one of them
also takes an astropy Quantity (``2.5 * u.mm``) or a string astropy parses
(``"2.5 mm"``, as in a configuration file), converted with `plasmacoll.units`.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Any, Literal, TypeAlias

import astropy.units as u
import numpy as np
from plasmacoll.units import (
    convert_units,
    to_unit,
    unit_field,
    with_unit,
)

ConfigDict: TypeAlias = dict[str, Any]

FaceType = Literal["dirichlet", "neumann"]

#: The 2D objects an electrode can be, by configuration-file ``type``.
OBJECT_TYPES = ("line", "rectangle", "triangle", "polygon", "circle", "aperture2d")


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


@dataclass
class GridParams:
    """The 2D grid of nodes: nodes per axis and extent in metres.

    ``coordinates`` is ``"cartesian"`` (planar, per unit depth) or ``"cylindrical"``
    (``(x, r)`` with the axis at ``r = 0``). A scalar ``num_gridpoints`` or bound
    applies to both axes; the bounds are lengths in m, or Quantities or strings such
    as ``"5 mm"`` (also per axis, ``("0 mm", "5 mm")``).
    ``ndim`` is always 2 (kept so that picnic's grid entries read unchanged).
    """

    ndim: int = 2
    num_gridpoints: Any = 64
    lower_bound: Any = unit_field("m", 0.0)
    upper_bound: Any = unit_field("m", 32.0e-3)
    coordinates: str = "cartesian"

    def __post_init__(self) -> None:
        """Convert the bounds to metres, normalize every per-axis value to a tuple."""
        convert_units(self)
        if int(self.ndim) != 2:
            raise ValueError(f"the beam tracer needs a 2D grid, got ndim={self.ndim}")
        self.ndim = 2
        self.num_gridpoints = tuple(int(n) for n in _per_axis(self.num_gridpoints))
        self.lower_bound = _per_axis(self.lower_bound)
        self.upper_bound = _per_axis(self.upper_bound)
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
            "lower_bound": with_unit(self.lower_bound, "m"),
            "upper_bound": with_unit(self.upper_bound, "m"),
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
    Expressions are parsed with sympy. ``expression`` may also be an astropy Quantity,
    e.g. ``5 * u.kV``: a constant, converted to ``unit`` (or kept in its own unit
    without one).
    """

    def __init__(
        self,
        expression: float | str,
        unit: str | None = None,
        variables: Mapping[str, Any] | None = None,
    ) -> None:
        """Parse the expression and its units."""
        if isinstance(expression, u.Quantity):
            if unit is None:
                unit = str(expression.unit)
            expression = to_unit(expression, unit, "value")
        self._expression = expression
        self._unit = unit
        self._variables = None if variables is None else dict(variables)
        self._factor = 1.0 if not unit else float(u.Quantity(1.0, unit).si.value)
        self._si_factors = {"x": 1.0, "y": 1.0, "z": 1.0, "t": 1.0}
        for name, variable in (self._variables or {}).items():
            if name not in self._si_factors:
                raise ValueError(f"unknown variable {name!r} (x, y, z, t)")
            self._si_factors[name] = 1.0 / float(
                u.Quantity(1.0, variable["unit"]).si.value
            )
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
            return {"value": with_unit(float(self._expression), self._unit or "")}
        func: ConfigDict = {"expression": str(self._expression), "unit": self._unit}
        if self._variables is not None:
            func["variables"] = self._variables
        return {"func": func}

    @classmethod
    def from_dict(
        cls, params: Mapping[str, Any], unit: str = "V"
    ) -> SymbolicSpaceFunction:
        """Create the value from ``{"value": ...}`` or ``{"func": {...}}``.

        A value is a number in ``unit`` or a quantity such as ``"5 kV"``, converted
        to ``unit``.
        """
        if "value" in params:
            if "unit" in params:
                raise ValueError(
                    "write the value of a field boundary with its unit, e.g. "
                    "value: 5 kV"
                )
            return cls(to_unit(params["value"], unit, "value"), unit)
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
        """Return a Dirichlet condition: a potential or an expression of x, y.

        ``value`` is a number or an expression in ``unit``, or a Quantity such as
        ``5 * u.kV``.
        """
        return FieldBoundaryCondition(
            "dirichlet", SymbolicSpaceFunction(value, unit, variables)
        )

    @staticmethod
    def neumann(
        normal_derivative: float | str = 0.0,
        unit: str = "V/m",
        variables: Mapping | None = None,
    ) -> FieldBoundaryCondition:
        """Return a Neumann condition on the normal derivative of the potential.

        ``normal_derivative`` is a number or an expression in ``unit``, or a Quantity
        such as ``1 * u.kV / u.mm``.
        """
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
        """Return the configuration-file form, the inverse of `from_dict`."""
        params: ConfigDict = {}
        for axis, sides in self._bc.items():
            lower = sides["min"]
            entry: ConfigDict = {"type": lower.bc_type}
            for key, bc in sides.items():
                side = bc.value.to_dict()
                if bc.bc_type != lower.bc_type:
                    side["type"] = bc.bc_type
                entry[key] = side["value"] if list(side) == ["value"] else side
            params[axis] = entry
        return params

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> FieldBCs:
        """Create the faces from ``{axis: {"type", "min": ..., "max": ...}}``.

        A side is a value (``5 kV``, or a number in V, V/m for Neumann),
        ``{value: ..., type: ...}`` for a side of another type, or
        ``{func: {expression, unit, variables}}`` for an expression of x and y::

            x: {type: dirichlet, min: 0 V, max: 5 kV}
        """
        faces: dict[str, FieldBoundaryCondition] = {}
        for axis in ("x", "y"):
            if axis not in params:
                raise ValueError(f"field boundaries need the {axis!r} axis")
            item = params[axis]
            axis_type = item.get("type")
            for key, name in (("min", "lower"), ("max", "upper")):
                side = item[key]
                if not isinstance(side, Mapping):
                    side = {"value": side}  # a bare value, e.g. ``min: 5 kV``
                bc_type = side.get("type", axis_type)
                if bc_type is None:
                    raise ValueError(f"field boundary '{axis}/{key}' needs a type")
                unit = "V/m" if bc_type == "neumann" else "V"
                faces[f"{axis}_{name}"] = FieldBoundaryCondition(
                    bc_type, SymbolicSpaceFunction.from_dict(side, unit)
                )
        return cls(**faces)


#: The geometry entries each shape type takes (simgeometry's constructor arguments).
_OBJECT_ARGUMENTS = {
    "line": ("midpoint", "side_lengths"),
    "rectangle": ("midpoint", "side_lengths"),
    "triangle": ("vertices",),
    "polygon": ("vertices",),
    "circle": ("midpoint", "radius"),
    "aperture2d": ("axis", "position", "vertices"),
}

#: The geometry entries of a shape, in the order of `ShapeParams`.
_GEOMETRY = ("vertices", "midpoint", "side_lengths", "radius", "axis", "position")


@dataclass
class ShapeParams:
    """One simgeometry shape of an electrode, without a potential.

    ``type`` is one of `OBJECT_TYPES`; the geometry entries are those of the
    simgeometry class, lengths in m (or Quantities, or strings such as
    ``"2.5 mm"``). An electrode of several shapes, `ObjectParams.union`, lists
    them.
    """

    type: str
    vertices: Any = unit_field("m", None)
    midpoint: Any = unit_field("m", None)
    side_lengths: Any = unit_field("m", None)
    radius: float | None = unit_field("m", None)
    axis: str | None = None
    position: float | None = unit_field("m", None)

    def __post_init__(self) -> None:
        """Convert the lengths to metres and check the type."""
        convert_units(self)
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
        """Return the configuration-file form (lengths in m)."""
        params: ConfigDict = {"type": self.type}
        if self.axis is not None:
            params["axis"] = self.axis
        for key in ("vertices", "midpoint", "side_lengths", "radius", "position"):
            value = getattr(self, key)
            if value is not None:
                params[key] = with_unit(value, "m")
        return params

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> ShapeParams:
        """Create the shape from its configuration-file entry (``bc`` is ignored)."""
        return cls(
            type=params["type"],
            axis=params.get("axis"),
            **{name: params.get(name) for name in _GEOMETRY if name != "axis"},
        )


@dataclass
class ObjectParams:
    """An electrode: a simgeometry object held at a potential (``bc_type="dirichlet"``).

    ``type`` is one of `OBJECT_TYPES`; the geometry entries are those of the
    simgeometry class, lengths in m. ``bc_value`` is the potential in V (the
    normal derivative in V/m for ``bc_type="neumann"``). Each takes a Quantity or a
    string such as ``"2.5 mm"`` or ``"5 kV"`` too. ``type="union"`` is an electrode of several
    shapes at one potential, the `ShapeParams` (or their configuration-file
    entries) in ``shapes``, e.g. a tube with rounded ends from a rectangle and two
    circles; see `union`. It is one electrode for the field, the hits and the
    diagnostics.
    """

    type: str
    bc_type: str
    bc_value: float | None = None
    vertices: Any = unit_field("m", None)
    midpoint: Any = unit_field("m", None)
    side_lengths: Any = unit_field("m", None)
    radius: float | None = unit_field("m", None)
    axis: str | None = None
    position: float | None = unit_field("m", None)
    shapes: list[ShapeParams] | None = None

    def __post_init__(self) -> None:
        """Convert the lengths and the potential, check the type, read the shapes."""
        convert_units(self)
        self.bc_value = to_unit(self.bc_value, self.bc_unit, "bc_value")
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

    @property
    def bc_unit(self) -> str:
        """Return the unit of ``bc_value``: V/m for Neumann, V otherwise."""
        return "V/m" if self.bc_type == "neumann" else "V"

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

        A union has none: build it with `build`.
        """
        if self.type == "union":
            raise ValueError("a union has no constructor arguments; use build()")
        arguments = self.shape_list[0].arguments()
        arguments["bc"] = self.bc_value
        return arguments

    def to_dict(self) -> ConfigDict:
        """Return the configuration-file form (lengths in m, the potential in V)."""
        if self.shapes is not None:
            params: ConfigDict = {
                "type": self.type,
                "shapes": [shape.to_dict() for shape in self.shapes],
            }
        else:
            params = self.shape_list[0].to_dict()
        key = "normal_derivative" if self.bc_type == "neumann" else "potential"
        params["bc"] = {
            "type": self.bc_type,
            key: with_unit(self.bc_value, self.bc_unit),
        }
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
        bc_value = params["bc"][key]
        if params["type"] == "union":
            return cls.union(
                [ShapeParams.from_dict(shape) for shape in params["shapes"]],
                bc_value,
                bc_type,
            )
        shape = ShapeParams.from_dict(params)
        geometry = {name: getattr(shape, name) for name in _GEOMETRY}
        return cls(type=shape.type, bc_type=bc_type, bc_value=bc_value, **geometry)
