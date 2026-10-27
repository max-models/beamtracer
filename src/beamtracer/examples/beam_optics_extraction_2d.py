r"""Beam optics of a proton extraction system: self-consistent trajectories.

The `BeamOptics` model solves the steady state of an extracted beam self-consistently:
it launches 200 representative proton trajectories ($1\,\mathrm{mA/m}$ per unit depth in this
planar geometry, at $33\,\mathrm{eV}$) from the plasma side, deposits the beam's space charge
along them, solves Poisson's equation with the electrodes, and repeats with relaxation (a
Vlasov iteration) until the field and the trajectories agree. The geometry is a grounded plasma
grid (PG) and an extraction grid (EG) at $-80\,\mathrm{V}$, with $-100\,\mathrm{V}$ and a small
cosine ripple on the downstream face.

The figures show the potential with the converged trajectories, the beam's space-charge density
and how the iteration converged.

Install::

    pip install "beamtracer[examples] @ git+https://github.com/max-models/beamtracer"
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import atomic_mass, e, m_p

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
    SymbolicSpaceFunction,
)


def extraction_grid_2d(
    *,
    axis: int = 0,
    position: float = 0.010,
    voltage: float = -80.0,
) -> ObjectParams:
    """Return a typical 2D extraction-grid aperture.

    A knife edge: a triangle with its tip at the aperture edge.
    """
    return ObjectParams(
        type="aperture2d",
        axis=axis,
        position=position,
        vertices=((0.030, 0.010), (0.033, 0.004), (0.036, 0.010)),
        bc_type="dirichlet",
        bc_value=voltage,
    )


def plasma_grid_2d(
    *,
    axis: int = 0,
    position: float = 0.010,
    voltage: float = 0.0,
) -> ObjectParams:
    """Return a typical 2D plasma-grid aperture."""
    return ObjectParams(
        type="aperture2d",
        axis=axis,
        position=position,
        vertices=((0.014, 0.010), (0.016, 0.003), (0.018, 0.010)),
        bc_type="dirichlet",
        bc_value=voltage,
    )


#: The proton beam: 1 mA per metre of depth from a 20 mm line, at 8e4 m/s (33 eV).
BEAM_LENGTH = 0.020
BEAM_ENERGY = 0.5 * m_p * 8.0e4**2 / e


def proton_beam(num_particles: int = 200) -> BeamSpecies:
    """Return the beam of the example."""
    return BeamSpecies(
        "H+", 1.0e-3 / BEAM_LENGTH, (0.0, 0.0), (0.0, BEAM_LENGTH), charge=1.0,
        mass=m_p / atomic_mass, energy=BEAM_ENERGY, parallel_temperature=0.0,
        transverse_temperature=0.0, num_particles=num_particles,
    )  # fmt: skip


def create_config(*, num_particles: int = 200, **settings: Any) -> BeamOpticsConfig:
    """Return the configuration of the ion extraction example.

    ``settings`` override fields of :class:`BeamOpticsConfig` (for example
    ``iterations=2``).
    """
    neumann = FieldBoundaryCondition.neumann(0.0)
    defaults = {
        "grid": GridParams(
            ndim=2,
            num_gridpoints=(81, 41),
            lower_bound=(0.0, 0.0),
            upper_bound=(0.040, 0.020),
            coordinates="cartesian",
        ),
        "bcs_field": FieldBCs(
            x_lower=FieldBoundaryCondition(
                bc_type="dirichlet",
                value=SymbolicSpaceFunction(expression="0.0", unit="V"),
            ),
            x_upper=FieldBoundaryCondition(
                bc_type="dirichlet",
                value=SymbolicSpaceFunction(
                    expression="cos((y / 20) * 2 * pi) * 5 - 100",
                    unit="V",
                    variables={"y": {"unit": "mm"}},
                ),
            ),
            y_lower=neumann,
            y_upper=neumann,
        ),
        "objects": {
            "pg": plasma_grid_2d(voltage=0.0),
            "eg": extraction_grid_2d(voltage=-80.0),
        },
        "beams": [proton_beam(num_particles)],
        "iterations": 20,
        "tolerance": 1.0e-3,
        "relaxation": 0.7,
        "save_trajectories": 4,
    }
    return BeamOpticsConfig(**{**defaults, **settings})


def create_model(**settings: Any) -> BeamOptics:
    """Create the ion extraction example model (see :func:`create_config`).

    Run it with :meth:`~beamtracer.model.BeamOptics.run`; its
    :meth:`~beamtracer.model.BeamOptics.write_files` writes the end-of-run file.
    """
    return BeamOptics(create_config(**settings))


STEM = Path(__file__).stem.replace("_", "-")


EXAMPLE = {
    "category": "Ion sources and extraction",
    "tags": ["BeamOptics", "2D", "Vlasov iteration", "space charge", "trajectories"],
    "runtime": "≈ 30 s",
    "depends": [
        "src/beamtracer/model.py",
        "src/beamtracer/solver/*.py",
        "src/beamtracer/config.py",
        "src/beamtracer/sections.py",
    ],
    "figures": {
        "": "Potential with the converged proton trajectories and the electrodes.",
        "charge": "Space-charge density of the beam deposited along the trajectories.",
        "convergence": "Change of the potential between Vlasov iterations.",
    },
}


def simulate():
    """Solve the example; return its fields, trajectories and convergence."""
    model = create_model().run()
    print(model.stats)
    return {
        "potential": model.potential_field.evaluate(name="potential", units="V"),
        "charge": model.charge_density.evaluate(name="charge density", units="C/m^3"),
        "trajectories": [np.asarray(t) for t in model.trajectories],
        "convergence": np.array(
            [h["max_potential_change"] for h in model.convergence_history[1:]]
        ),
        "stats": dict(model.stats),
    }


def millimetres(array):
    """Return a field with its ``x`` and ``y`` coordinates in millimetres."""
    array = array.assign_coords(x=array.x * 1e3, y=array.y * 1e3)
    array.x.attrs.update(long_name="x", units="mm")
    array.y.attrs.update(long_name="y", units="mm")
    return array


def joined(paths):
    """The paths as one NaN-separated line (in mm), for a ``lines`` overlay."""
    pieces = []
    for path in paths:
        pieces += [path * 1e3, np.full((1, 2), np.nan)]
    line = np.vstack(pieces[:-1])
    return line[:, 0], line[:, 1]


#: Formats a gallery figure is saved in by default (``.png`` needs kaleido and Chrome).
DEFAULT_FORMATS = ("plotly.json", "png")


def save(
    figure: Any,
    name: str,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
    show: bool = False,
    **kwargs: Any,
) -> None:
    """Save a gallery figure as ``<name>.<format>`` (see :func:`plasma_plots.save_figure`).

    A ``png`` format is skipped with a note when kaleido cannot export images.
    """
    from plasma_plots import save_figure

    formats = list(formats)
    if "png" in formats:
        try:
            import kaleido  # noqa: F401
        except ImportError:
            formats.remove("png")
            print(f"{name}: kaleido is not installed, no PNG written")
    save_figure(figure, name, formats=tuple(formats), show=show, **kwargs)


def object_outlines(config: Any) -> dict[str, tuple[Any, Any]]:
    """Return the closed outlines of a 2D configuration's objects, ``{name: (x, y)}``.

    An object made of several polygons (an aperture's two plates) is one line, its polygons
    separated by NaN. The format is that of the ``lines`` overlay of plasma-plots'
    ``.plasma.plot.slice``.
    """
    import numpy as np

    outlines = {}
    for name, params in config.objects.items():
        obj = params.build()
        # close each loop, but not an open curve (a simgeometry Polyline)
        closed = getattr(obj, "closed", True)
        pieces = []
        for loop in obj.outlines():
            vertices = loop.T
            end = vertices[:1] if closed else np.zeros((0, 2))
            pieces += [vertices, end, np.full((1, 2), np.nan)]
        line = np.vstack(pieces[:-1])
        outlines[name] = (line[:, 0], line[:, 1])
    return outlines


def pproc(result, show=False, formats=("plotly.json", "png")):
    """Draw and save the figures of a result of :func:`simulate`."""
    import plasma_plots

    outlines = {
        {"pg": "plasma grid", "eg": "extraction grid"}[name]: (x * 1e3, y * 1e3)
        for name, (x, y) in object_outlines(create_config()).items()
    }
    stats = result["stats"]
    with plasma_plots.figure(1, 1, figsize=(8.0, 4.6), backend="plotly") as figure:
        ax = figure[0]
        millimetres(result["potential"]).plasma.plot.slice(
            x="x",
            y="y",
            cmap="viridis",
            title=f"Potential and trajectories: {stats['transmitted']} of "
            f"{stats['launched']} transmitted",
            overlays={"lines": outlines},
            ax=ax,
        )
        x, y = joined(result["trajectories"])
        ax.plot(x, y, color="white", lw=0.8, alpha=0.8, label="trajectories")
        ax.legend(loc="upper right")
    save(figure, STEM, formats=formats, show=show)

    figure = millimetres(result["charge"]).plasma.plot.slice(
        x="x",
        y="y",
        cmap="magma",
        title="Space charge of the beam",
        overlays={"lines": outlines},
        backend="plotly",
    )
    save(figure, f"{STEM}-charge", formats=formats, show=show)

    history = result["convergence"]
    with plasma_plots.figure(1, 1, backend="plotly") as figure:
        ax = figure[0]
        if history.ndim > 1:  # one column per recorded quantity: plot the first
            history = history[:, 0]
        ax.semilogy(np.arange(1, len(history) + 1), np.abs(history), "o-")
        ax.set_xlabel("Vlasov iteration")
        ax.set_ylabel("change of the potential")
        ax.set_title("Convergence of the self-consistent beam")
    save(figure, f"{STEM}-convergence", formats=formats, show=show)


def run_directory(stem: str) -> Path:
    """Return the directory a gallery script keeps its run output in."""
    return Path("runs") / stem


def _keep(result: Any, store: Path) -> None:
    """Keep the result for ``--pproc-only``."""
    store.parent.mkdir(parents=True, exist_ok=True)
    try:
        store.write_bytes(pickle.dumps(result))
    except (pickle.PicklingError, TypeError, AttributeError):
        pass


def _load(stem: str, store: Path) -> Any:
    if store.exists():
        return pickle.loads(store.read_bytes())
    directory = run_directory(stem)
    raise SystemExit(
        f"no previous run of {stem} in {directory}: run without --pproc-only first"
    )


def main(
    script: str,
    simulate: Callable[[], Any],
    pproc: Callable[..., None],
    argv: Sequence[str] | None = None,
) -> None:
    """Run a gallery script: simulate (or reload) the example, then draw its figures."""
    stem = Path(script).stem.replace("_", "-")
    parser = argparse.ArgumentParser(
        description=f"Run the {stem} example and save its figures."
    )
    parser.add_argument(
        "--pproc-only",
        action="store_true",
        help="redraw the figures of the previous run",
    )
    parser.add_argument(
        "--show", action="store_true", help="show the figures before saving them"
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated figure formats (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    formats = tuple(fmt.strip() for fmt in args.formats.split(",") if fmt.strip())
    store = run_directory(stem) / "result.pkl"
    if args.pproc_only:
        result = _load(stem, store)
    else:
        result = simulate()
        _keep(result, store)
    pproc(result, show=args.show, formats=formats)


if __name__ == "__main__":
    main(__file__, simulate, pproc)
