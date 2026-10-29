r"""A neutralizer channel: the beam traced through the plasma it makes.

A 1 MeV, 10 A deuterium negative-ion beam crosses one channel of a beam-driven plasma
neutralizer, 3 m long and 10 cm wide. On the gas, and on the plasma it makes itself,
the beam changes charge state; the plasma is a global model of the channel
(`wellbalanced`), fed by the beam's stripped and secondary electrons. Alone, that
model treats the beam as a line integral along the axis. Here the beam is traced
instead (`beamtracer.neutralizer`): `BeamOptics` trajectories through the channel, each
carrying its charge-state fractions on the gas and the plasma, and the plasma takes
its sources from the traced paths; the two are iterated to a fixed point.

With a parallel beam the traced result is the axis model's. As the beam diverges,
more of it hits the channel's walls before the exit: at 10 mrad a fifth of it, and the
neutral fraction falls from 0.57 to 0.47; the plasma gets fewer sources. The neutral
share of the beam that leaves stays at 0.57: every transmitted particle crosses the
same uniform gas and plasma over the same length, and the plasma's stripping is
small, so the loss is all geometry. The neutrals that do leave diverge less than the
beam entered (8 mrad at 10), since the wide angles are the ones lost. The cross sections and rates are
illustrative placeholders, not data
(`wellbalanced.neutralizer.illustrative_deuterium_neutralizer`): the figures show
what the coupled models compute, not what a neutralizer achieves.

Install::

    pip install "beamtracer[examples] @ git+https://github.com/max-models/beamtracer"
    pip install "wellbalanced[cr] @ git+https://github.com/max-models/wellbalanced@devel"
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from beamtracer.neutralizer import NeutralizerChannel

#: rms divergence of the beam at the entrance (rad).
DIVERGENCES = (0.0, 0.003, 0.006, 0.01, 0.015)
#: Beam width at the entrance (m), in the 10 cm channel.
BEAM_WIDTH = 0.08
#: Gas molecules fed per second: near the gas's best neutral fraction.
GAS_INFLOW = 6.0e21

STEM = Path(__file__).stem.replace("_", "-")

EXAMPLE = {
    "category": "Ion sources and extraction",
    "tags": ["BeamOptics", "2D", "neutralizer", "charge exchange", "wellbalanced"],
    "runtime": "≈ 4 min",
    "depends": [
        "src/beamtracer/neutralizer.py",
        "src/beamtracer/solver/charge_changing.py",
        "src/beamtracer/solver/tracer.py",
        "src/beamtracer/model.py",
    ],
    "figures": {
        "": "Beam trajectories through the channel at 10 mrad divergence.",
        "divergence": "Transmission and neutral fraction against the beam's divergence, traced and along the axis.",
        "states": "Charge-state fractions along the channel (axis model) and at the exit (traced).",
    },
}


def neutralizer():
    """The channel with OpenCR's plasma rates (the beam data are placeholders, which warn)."""
    from wellbalanced.cr import deuterium_neutralizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return deuterium_neutralizer(gas_inflow=GAS_INFLOW)


def channel(divergence: float) -> NeutralizerChannel:
    """Return the traced channel for a beam of rms ``divergence`` [rad]."""
    return NeutralizerChannel(
        neutralizer(), beam_width=BEAM_WIDTH, divergence=divergence, num_particles=200
    )


def create_config():
    """Return the beam configuration of the channel at 10 mrad."""
    return channel(0.01).beam_config(1.0e19, 1.0e17)


def simulate():
    """Solve the channel at every divergence of the scan."""
    runs = []
    for divergence in DIVERGENCES:
        result = channel(divergence).solve()
        runs.append(
            {
                "divergence": divergence,
                "neutral": result.neutral_fraction,
                "axis_neutral": result.axis.neutral_fraction,
                "transmission": result.transmission,
                "exit": result.exit_fractions,
                "neutral_divergence": result.neutral_divergence,
                "electron_density": result.plasma.electron_density,
                "trajectories": [np.asarray(t) for t in result.model.trajectories],
                "axis_position": result.axis.position,
                "axis_fractions": result.axis.fractions,
                "converged": result.converged,
                "iterations": result.iterations,
            }
        )
        print(
            f"divergence {divergence * 1e3:4.1f} mrad: neutral fraction "
            f"{result.neutral_fraction:.4f} (axis {result.axis.neutral_fraction:.4f}), "
            f"transmission {result.transmission:.3f}, neutrals leave at "
            f"{result.neutral_divergence * 1e3:.2f} mrad; n_e "
            f"{result.plasma.electron_density:.3e} m^-3; "
            f"{'converged' if result.converged else 'NOT converged'} in "
            f"{result.iterations} iterations"
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


def pproc(result, show=False, formats=("plotly.json", "png")):
    """Draw and save the figures of a result of :func:`simulate`."""
    import plotly.graph_objects as go

    runs = result["runs"]
    shown = min(runs, key=lambda run: abs(run["divergence"] - 0.01))
    figure = go.Figure()
    half = 0.05
    for y in (-half, half):
        figure.add_trace(
            go.Scatter(x=[0.0, 3.0], y=[1e3 * y] * 2, mode="lines",
                       line={"color": "black", "width": 3}, showlegend=False)
        )  # fmt: skip
    for k, path in enumerate(shown["trajectories"]):
        figure.add_trace(
            go.Scatter(
                x=path[:, 0],
                y=1e3 * path[:, 1],
                mode="lines",
                line={"color": "#2a78d6", "width": 1},
                name="trajectories",
                showlegend=k == 0,
            )
        )
    figure.update_layout(
        title=f"The beam through the channel, {shown['divergence'] * 1e3:.0f} mrad rms",
        xaxis={"title": "x (m)"},
        yaxis={"title": "y (mm)"},
        template="plotly_white",
    )
    save(figure, STEM, formats=formats, show=show)

    divergence = [1e3 * run["divergence"] for run in runs]
    figure = go.Figure()
    curves = (
        ("transmission (all states)", [run["transmission"] for run in runs], "dot"),
        ("neutral fraction, traced", [run["neutral"] for run in runs], "solid"),
        (
            "neutral fraction, along the axis",
            [run["axis_neutral"] for run in runs],
            "dash",
        ),
        (
            "neutrals in the transmitted beam",
            [run["neutral"] / run["transmission"] for run in runs],
            "dashdot",
        ),
    )
    for name, values, dash in curves:
        figure.add_trace(
            go.Scatter(x=divergence, y=values, mode="lines+markers", name=name,
                       line={"dash": dash})
        )  # fmt: skip
    figure.update_layout(
        title="What the walls take: the neutrals in the beam that leaves stay the same",
        xaxis={"title": "rms divergence at the entrance (mrad)"},
        yaxis={"title": "share of the incoming particles", "range": [0.0, 1.05]},
        template="plotly_white",
    )
    save(figure, f"{STEM}-divergence", formats=formats, show=show)

    figure = go.Figure()
    first = runs[0]
    for k, state in enumerate(("-", "0", "+")):
        figure.add_trace(
            go.Scatter(x=first["axis_position"], y=first["axis_fractions"][:, k],
                       mode="lines", name=f"{state}, along the axis")
        )  # fmt: skip
        figure.add_trace(
            go.Scatter(x=[3.0] * len(runs), y=[run["exit"][state] for run in runs],
                       mode="markers", name=f"{state} at the exit, traced",
                       text=[f"{run['divergence'] * 1e3:.0f} mrad" for run in runs])
        )  # fmt: skip
    figure.update_layout(
        title="Charge states along the channel",
        xaxis={"title": "x (m)"},
        yaxis={"title": "fraction of the incoming particles"},
        template="plotly_white",
    )
    save(figure, f"{STEM}-states", formats=formats, show=show)


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
