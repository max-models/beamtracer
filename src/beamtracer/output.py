"""Results on the nodes of the grid.

:class:`NodeField` holds a field on the nodes of a 2D grid (one or several components,
the component axis last) and evaluates it as an :class:`xarray.DataArray`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class NodeField:
    """A field on the nodes of a 2D grid from ``lower_bound`` to ``upper_bound``.

    ``data`` has the shape ``(n0, n1)`` or ``(n0, n1, num_components)``; the nodes are
    ``linspace(lower, upper, n)`` along each axis.
    """

    def __init__(
        self,
        data: np.ndarray,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
    ) -> None:
        """Store the data (with its component axis last) and the extent."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 2:
            data = data[..., None]
        if data.ndim != 3:
            raise ValueError(
                f"node data must be 2D (plus components), got {data.shape}"
            )
        self._data = data
        self.lower_bound = tuple(float(v) for v in lower_bound)
        self.upper_bound = tuple(float(v) for v in upper_bound)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
    ) -> NodeField:
        """Return the field of the node values ``data``."""
        return cls(data, lower_bound, upper_bound)

    @property
    def num_components(self) -> int:
        """Return the number of components."""
        return int(self._data.shape[-1])

    @property
    def num_gridpoints(self) -> tuple[int, int]:
        """Return the number of nodes along each axis."""
        return int(self._data.shape[0]), int(self._data.shape[1])

    def coordinates(self, axis: int) -> np.ndarray:
        """Return the node coordinates along ``axis`` (0 or 1)."""
        return np.linspace(
            self.lower_bound[axis], self.upper_bound[axis], self._data.shape[axis]
        )

    def get_field_data(self, component: int | None = None) -> np.ndarray:
        """Return the node values: of one component, or all with the component axis last."""
        if component is None:
            return self._data if self.num_components > 1 else self._data[..., 0]
        return self._data[..., component]

    def to_numpy(self) -> np.ndarray:
        """Return the node values, shape ``(n0, n1, num_components)``."""
        return self._data

    def evaluate(
        self,
        x: Any = None,
        y: Any = None,
        component: int | None = None,
        name: str | None = None,
        units: str | None = None,
    ) -> Any:
        """Return the field as an ``xarray.DataArray`` with ``x`` and ``y`` dims.

        An omitted axis keeps the nodes; a scalar or a sequence of coordinates on an axis
        interpolates linearly there (a scalar drops the dimension). A field with several
        components has a ``component`` dimension unless one is chosen. ``name`` and
        ``units`` label the result.
        """
        import xarray as xr

        if component is None and self.num_components == 1:
            component = 0
        values = self._data if component is None else self._data[..., component]
        dims = ["x", "y"] + ([] if component is not None else ["component"])
        array = xr.DataArray(
            values,
            dims=dims,
            coords={"x": self.coordinates(0), "y": self.coordinates(1)},
            name=name,
        )
        selection = {
            key: value for key, value in (("x", x), ("y", y)) if value is not None
        }
        if selection:
            array = array.interp(selection)
        if units is not None:
            array.attrs["units"] = units
        return array
