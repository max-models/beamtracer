r"""Steady negative-ion extraction: an H$^-$ beamlet and its electrons in one solve.

The `BeamOptics` model with `NegativeExtractionPlasma` solves a beamlet of negative ions in
its own field, as is usual for positive ions: the H$^-$ ions and the electrons extracted
with them are traced from the plasma, their space charge is deposited along the
trajectories, and the plasma's positive ions, repelled by the extraction field, follow a
Boltzmann distribution, $\rho_p \exp(-(\phi - U_p)/T_p)$, which forms the meniscus. A Vlasov
iteration repeats the solve until the field and the trajectories agree. One solve takes one
to three minutes on one core, where a time-dependent PIC run of a beamlet takes tens of
minutes to settle: this is the evaluation an optimisation of the grids would repeat many
times.

The beamlet is round ($r$-$z$): a plasma grid (PG, 0 V) with an 8 mm aperture and a chamfer,
an extraction grid (EG) and a grounded grid at $20\,\mathrm{kV}$ (GG), 300 A/m$^2$ of H$^-$
and as much electron current. The geometry is illustrative, not a design. The extraction
voltage is scanned at fixed current. From 10 to 4 kV, i.e. from 0.14 to 0.54 of the Child-
Langmuir current of the extraction gap, the rms divergence behind the GG falls from 41 to 16
mrad: at high voltage the meniscus is concave and over-focuses the beam. At 5 and 6 kV the
beam grazes the EG (0.7 to 2.4% of the H$^-$ is lost there) and the iteration does not
settle: the potential keeps changing by a few volts per iteration, with stronger relaxation
or three times the trajectories too, and the divergence depends on where it stops (23 to 27
mrad at 5 kV). An optimisation has to treat such points as unconverged; the scan figure
marks them.

Without a magnetic field the electrons take nearly the paths of the H$^-$: in a static
electric field a trajectory depends on the charge and the energy, not the mass, and the two
differ only in their thermal spread at the start. Dumping them on the EG needs deflection
magnets, and this model has solenoid fields only. The electron current is an input here, not
a result: the ratio of electrons to H$^-$ is set by the source plasma and needs a PIC model
of the source.

Install::

    pip install "beamtracer[examples] @ git+https://github.com/max-models/beamtracer"
"""

from __future__ import annotations

import argparse
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import constants

from beamtracer import (
    BeamOptics,
    BeamOpticsConfig,
    BeamSpecies,
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    NegativeExtractionPlasma,
    ObjectParams,
)

H_MINUS_MASS = 1.00849  # u


@dataclass
class BeamletCase:
    """One round beamlet: grids, voltages, beams and numerics (lengths in m, voltages in V)."""

    #: Aperture radius of the plasma grid, of the extraction grid, of the grounded grid.
    pg_radius: float = 4.0e-3
    eg_radius: float = 3.0e-3
    gg_radius: float = 3.5e-3
    #: ``(front, back)`` of each grid along the axis.
    pg: tuple[float, float] = (1.0e-3, 2.0e-3)
    eg: tuple[float, float] = (6.0e-3, 8.0e-3)
    gg: tuple[float, float] = (16.0e-3, 18.0e-3)
    extraction_voltage: float = 5.0e3
    grounded_grid_voltage: float = 20.0e3
    plasma_potential: float = -1.0
    #: H- current density [A/m^2] and electrons per H- (current).
    current_density: float = 300.0
    electron_ratio: float = 1.0
    #: Starting energy and transverse temperature of the H- and of the electrons [eV].
    h_minus_energy: float = 1.0
    h_minus_temperature: float = 1.0
    electron_energy: float = 1.0
    electron_temperature: float = 2.0
    ion_temperature: float = 1.0
    #: Mesh: spacing, length, outer radius; trajectories per species.
    cell: float = 0.1e-3
    length: float = 24.0e-3
    radius: float = 6.0e-3
    num_particles: int = 300
    iterations: int = 40
    tolerance: float = 1.0e-2
    relaxation: float = 0.5

    @property
    def exit_plane(self) -> float:
        """Return the diagnostic plane behind the grounded grid [m]."""
        return self.length - 0.5e-3

    @property
    def emission_radius(self) -> float:
        """The beams leave the plasma side over the PG aperture, one cell inside it."""
        return self.pg_radius - self.cell

    @property
    def child_langmuir(self) -> float:
        """Child-Langmuir current density [A/m^2] of H- over the PG-EG gap."""
        gap = self.eg[0] - self.pg[0]
        return (
            4.0 * constants.epsilon_0 / 9.0
            * np.sqrt(2.0 * constants.e / (H_MINUS_MASS * constants.atomic_mass))
            * self.extraction_voltage**1.5 / gap**2
        )  # fmt: skip


def _grid(
    name: str, span: tuple[float, float], radius: float, outer: float, voltage: float
):
    """A grid plate around the axis, from its aperture radius outwards."""
    x0, x1 = span
    vertices = ((x0, radius), (x0, outer), (x1, outer), (x1, radius))
    if name == "pg":  # a chamfer opening towards the extraction gap
        vertices = ((x0, radius), (x0, outer), (x1, outer), (x1, radius + (x1 - x0)))
    return ObjectParams(
        type="aperture2d",
        axis=0,
        position=0.0,
        vertices=vertices,
        bc_type="dirichlet",
        bc_value=voltage,
    )


def beamlet_config(case: BeamletCase | None = None) -> BeamOpticsConfig:
    """The ``BeamOptics`` configuration of ``case``."""
    case = case or BeamletCase()
    outer = case.radius + case.cell  # the plates reach past the outer face
    neumann = FieldBoundaryCondition.neumann(0.0)
    h_minus = BeamSpecies(
        "H-",
        case.current_density,
        (0.0, 0.0),
        (0.0, case.emission_radius),
        charge=-1.0,
        mass=H_MINUS_MASS,
        energy=case.h_minus_energy,
        transverse_temperature=case.h_minus_temperature,
        num_particles=case.num_particles,
    )
    beams = [h_minus]
    if case.electron_ratio > 0.0:
        beams.append(
            replace(
                h_minus,
                name="e",
                mass=constants.m_e / constants.atomic_mass,
                current_density=case.electron_ratio * case.current_density,
                energy=case.electron_energy,
                transverse_temperature=case.electron_temperature,
            )
        )
    return BeamOpticsConfig(
        grid=GridParams(
            ndim=2,
            num_gridpoints=(
                round(case.length / case.cell) + 1,
                round(case.radius / case.cell) + 1,
            ),
            lower_bound=(0.0, 0.0),
            upper_bound=(case.length, case.radius),
            coordinates="cylindrical",
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition.dirichlet(case.plasma_potential),
            x_upper=FieldBoundaryCondition.dirichlet(case.grounded_grid_voltage),
            y_lower=neumann,
            y_upper=neumann,
        ),
        objects={
            "pg": _grid("pg", case.pg, case.pg_radius, outer, 0.0),
            "eg": _grid("eg", case.eg, case.eg_radius, outer, case.extraction_voltage),
            "gg": _grid(
                "gg", case.gg, case.gg_radius, outer, case.grounded_grid_voltage
            ),
        },
        beams=beams,
        plasma=NegativeExtractionPlasma(
            case.ion_temperature, case.plasma_potential, case.pg[0]
        ),
        iterations=case.iterations,
        tolerance=case.tolerance,
        relaxation=case.relaxation,
        diagnostic_planes=[case.exit_plane],
        save_trajectories=10,
    )


def run_case(case: BeamletCase) -> dict[str, Any]:
    """Solve ``case``; return the H- beam behind the GG, the losses and the cost."""
    model = BeamOptics(beamlet_config(case))
    start = time.perf_counter()
    model.run()
    seconds = time.perf_counter() - start
    h_minus = model.beam_moments(case.exit_plane, "H-")
    emitted = np.pi * case.emission_radius**2 * case.current_density
    status = model.state.trace.status
    species = np.concatenate(
        [np.full(case.num_particles, k) for k in range(len(model.species_names))]
    )
    return {
        "extraction_voltage": case.extraction_voltage,
        "perveance_ratio": case.current_density / case.child_langmuir,
        "converged": bool(model.stats["converged"]),
        "iterations": model.stats["iterations"],
        "seconds": seconds,
        "transmitted": h_minus.current / emitted,
        "divergence": h_minus.rms_angle,
        "emittance": h_minus.emittance,
        "energy": h_minus.mean_energy,
        "lost_h_minus": int(np.sum((status != 0) & (species == 0))),
        "history": np.array(
            [h["max_potential_change"] for h in model.convergence_history[1:]]
        ),
        "potential": model.potential_field.evaluate(name="potential", units="V"),
        "trajectories": [np.asarray(t) for t in model.trajectories],
        "trajectory_species": np.asarray(model.trajectory_species),
    }


STEM = Path(__file__).stem.replace("_", "-")
CASE = BeamletCase()
#: Extraction voltages of the scan [V], at the case's current density.
SCAN_VOLTAGES = (4.0e3, 5.0e3, 6.0e3, 8.0e3, 10.0e3)

EXAMPLE = {
    "category": "Ion sources and extraction",
    "tags": [
        "BeamOptics",
        "2D",
        "r-z",
        "H-",
        "negative ions",
        "Vlasov iteration",
        "space charge",
    ],
    "runtime": "≈ 10 min",
    "depends": [
        "src/beamtracer/model.py",
        "src/beamtracer/solver/*.py",
        "src/beamtracer/config.py",
    ],
    "figures": {
        "": "Potential, grids and the H- and electron trajectories of one beamlet.",
        "scan": "Divergence and transmission of the H- against the extraction voltage.",
        "convergence": "Change of the potential between Vlasov iterations in the scan.",
    },
}


def create_config():
    """Return the configuration of the default beamlet."""
    return beamlet_config(CASE)


def simulate():
    """Solve the beamlet at every extraction voltage of the scan."""
    runs = []
    for voltage in SCAN_VOLTAGES:
        run = run_case(replace(CASE, extraction_voltage=voltage))
        runs.append(run)
        print(
            f"U_ex {voltage / 1e3:4.1f} kV (J/J_CL {run['perveance_ratio']:4.2f}): "
            f"{'converged' if run['converged'] else 'not converged'} in "
            f"{run['iterations']} iterations, {run['seconds']:5.1f} s; H- "
            f"{run['transmitted']:6.1%} transmitted, rms divergence "
            f"{run['divergence'] * 1e3:5.1f} mrad, emittance "
            f"{run['emittance'] * 1e6:5.2f} mm mrad at {run['energy'] / 1e3:4.1f} keV"
        )
    return {"runs": runs}


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


def millimetres(array):
    """Return a field with its ``x`` and ``y`` coordinates in millimetres."""
    array = array.assign_coords(x=array.x * 1e3, y=array.y * 1e3)
    array.x.attrs.update(long_name="x", units="mm")
    array.y.attrs.update(long_name="r", units="mm")
    return array


def joined(paths):
    """The paths as one NaN-separated line (in mm)."""
    if not paths:
        return np.zeros(0), np.zeros(0)
    pieces = []
    for path in paths:
        pieces += [path * 1e3, np.full((1, 2), np.nan)]
    line = np.vstack(pieces[:-1])
    return line[:, 0], line[:, 1]


def grid_outlines(case: BeamletCase) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The grids' cross sections above the axis [mm]."""
    config = beamlet_config(case)
    lines = {}
    for name, label in (("pg", "PG"), ("eg", "EG"), ("gg", "GG")):
        vertices = np.asarray(config.objects[name].vertices, dtype=float)
        closed = np.vstack([vertices, vertices[:1]]) * 1e3
        lines[label] = (closed[:, 0], np.minimum(closed[:, 1], case.radius * 1e3))
    return lines


def pproc(result, show=False, formats=("plotly.json", "png")):
    """Draw and save the figures of a result of :func:`simulate`."""
    import plasma_plots
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    runs = result["runs"]
    converged = [run for run in runs if run["converged"]] or runs
    best = min(converged, key=lambda run: run["divergence"])
    case = replace(CASE, extraction_voltage=best["extraction_voltage"])
    with plasma_plots.figure(1, 1, figsize=(9.0, 3.6), backend="plotly") as figure:
        ax = figure[0]
        millimetres(best["potential"]).plasma.plot.slice(
            x="x",
            y="y",
            cmap="viridis",
            title=f"U_ex = {best['extraction_voltage'] / 1e3:.0f} kV: "
            f"H- (white) and electrons (red, on nearly the same paths)",
            overlays={"lines": grid_outlines(case)},
            ax=ax,
        )
        species = best["trajectory_species"]
        paths = best["trajectories"]
        for k, color, name in ((1, "red", "electrons"), (0, "white", "H-")):
            x, y = joined([p for p, s in zip(paths, species, strict=True) if s == k])
            if x.size:
                ax.plot(x, y, color=color, lw=0.7, alpha=0.8, label=name)
        ax.legend(loc="upper right")
    save(figure, STEM, formats=formats, show=show)

    voltages = [run["extraction_voltage"] / 1e3 for run in runs]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=voltages,
            y=[run["divergence"] * 1e3 for run in runs],
            mode="lines+markers",
            name="rms divergence [mrad]",
            marker={
                "symbol": ["circle" if run["converged"] else "x" for run in runs],
                "size": 10,
            },
            text=[
                f"J/J_CL {run['perveance_ratio']:.2f}, "
                + ("converged" if run["converged"] else "not converged")
                for run in runs
            ],
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=voltages,
            y=[100 * run["transmitted"] for run in runs],
            mode="lines+markers",
            name="H- transmitted [%]",
            line={"dash": "dot"},
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="extraction voltage [kV]")
    figure.update_yaxes(
        title_text="rms divergence behind the GG [mrad]", secondary_y=False
    )
    figure.update_yaxes(title_text="transmitted [%]", secondary_y=True)
    figure.update_layout(
        title="300 A/m² of H-: divergence against the extraction voltage "
        "(x: not converged)",
        template="plotly_white",
    )
    save(figure, f"{STEM}-scan", formats=formats, show=show)

    figure = go.Figure()
    for run in runs:
        figure.add_trace(
            go.Scatter(
                x=np.arange(1, run["history"].size + 1),
                y=run["history"],
                mode="lines+markers",
                name=f"{run['extraction_voltage'] / 1e3:.0f} kV",
            )
        )
    figure.update_layout(
        title="Convergence of the Vlasov iteration",
        xaxis={"title": "iteration"},
        yaxis={"title": "largest change of the potential [V]", "type": "log"},
        template="plotly_white",
    )
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
    raise SystemExit(
        f"no previous run of {stem} in {store.parent}: run without --pproc-only first"
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
