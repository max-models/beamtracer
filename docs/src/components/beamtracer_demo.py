"""The live demo of the docs: ions focused by beamtracer's einzel lens.

It runs in the browser (Pyodide); `run` takes the values of the sliders and returns the
figure. One vacuum field solve and trace of `beamtracer.optimization.einzel.EinzelLens`
on a coarse mesh (1 mm), without space charge, and its thin-lens and paraxial foci.
"""

import math

import numpy as np

from beamtracer.optimization.einzel import EinzelLens

#: The colour of the electrodes, readable on light and dark pages.
ELECTRODE_COLOUR = "rgba(128, 128, 128, 0.55)"


def run(voltage=15.0, energy=30.0, beam_radius=2.0, aperture=5.0):
    """Trace a parallel proton beam through an einzel lens.

    Args:
        voltage: The voltage of the middle tube [kV]; positive decelerates the protons.
        energy: The energy of the protons [keV].
        beam_radius: The radius of the parallel beam at its start [mm].
        aperture: The bore radius of the three tubes [mm].

    Returns:
        The figure of the trajectories and electrodes, the foci and the spot as
        ``numbers``, and a ``text`` when no ion reaches the target or there is no lens.
    """
    lens = EinzelLens(
        energy=energy * 1e3,
        voltage=voltage * 1e3,
        beam_radius=beam_radius * 1e-3,
        aperture=aperture * 1e-3,
        cell=1e-3,
        num_particles=12,
        save_trajectories=1,
    )
    model = lens.solve()
    axis = model.potential[:, 0]
    thin = lens.thin_lens_focal_length(model.mesh.x, axis)
    paraxial = lens.paraxial_focus(model.mesh.x, axis)
    evaluation = lens.measure(model)
    transmission = evaluation.diagnostics["transmission"]

    # the rays [mm, to 1 um], mirrored at the axis, as one line broken by gaps
    x, r = [], []
    for points in model.trajectories:
        points = np.round(1e3 * points, 3).tolist()
        for sign in (1.0, -1.0):
            x.extend([p[0] for p in points] + [None])
            r.extend([sign * p[1] for p in points] + [None])
    data = [
        {
            "x": x,
            "y": r,
            "mode": "lines",
            "name": "ions",
            "line": {"width": 1.5},
            "hoverinfo": "skip",
        }
    ]
    numbers = {}
    for name, focus in (
        ("thin lens", lens.centre + thin),
        ("paraxial ray", paraxial),
    ):
        shown = math.isfinite(focus) and lens.voltage != 0.0 and abs(focus) < 10.0
        if shown and 0.0 < focus < 3.0 * lens.length:
            data.append(
                {
                    "x": [1e3 * focus],
                    "y": [0.0],
                    "mode": "markers",
                    "name": f"focus, {name}",
                    "marker": {
                        "size": 10,
                        "symbol": "x" if name == "thin lens" else "circle-open",
                    },
                }
            )
        numbers[f"Focus, {name} [mm]"] = 1e3 * focus if shown else "none"
    if transmission > 0.0:
        numbers["rms size at the target [mm]"] = 1e3 * evaluation.objectives["spot"]
    numbers["Transmission"] = f"{100.0 * transmission:.0f} %"

    # the three tubes, above and below the axis, and the target plane
    half = 0.5 * lens.electrode_length
    shapes = []
    for centre in lens.electrode_positions(lens.gap):
        for sign in (1.0, -1.0):
            inner, outer = sign * lens.aperture, sign * (lens.aperture + lens.thickness)
            shapes.append(
                {
                    "type": "rect",
                    "x0": 1e3 * (centre - half),
                    "x1": 1e3 * (centre + half),
                    "y0": 1e3 * inner,
                    "y1": 1e3 * outer,
                    "fillcolor": ELECTRODE_COLOUR,
                    "line": {"width": 0},
                }
            )
    shapes.append(
        {
            "type": "line",
            "x0": 1e3 * lens.target,
            "x1": 1e3 * lens.target,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"dash": "dash", "width": 1},
        }
    )
    annotations = [
        {
            "x": 1e3 * lens.centre,
            "y": 1e3 * (lens.aperture + lens.thickness),
            "text": f"{voltage:g} kV",
            "showarrow": False,
            "yanchor": "bottom",
        },
        {
            "x": 1e3 * lens.target,
            "y": 1,
            "yref": "paper",
            "text": "target",
            "showarrow": False,
            "xanchor": "right",
            "yanchor": "top",
        },
    ]
    extent = 1e3 * (lens.aperture + lens.thickness) + 1.0
    figure = {
        "data": data,
        "layout": {
            "xaxis": {"title": {"text": "x [mm]"}, "range": [0.0, 1e3 * lens.length]},
            "yaxis": {"title": {"text": "r [mm]"}, "range": [-extent, extent]},
            "shapes": shapes,
            "annotations": annotations,
            "legend": {"orientation": "h", "y": -0.25},
        },
    }
    result = {"figure": figure, "numbers": numbers}
    if lens.voltage == 0.0:
        result["text"] = "Without a voltage on the middle tube there is no lens."
    elif math.isnan(thin):
        result["text"] = "The protons are stopped by the middle tube and turn back."
    elif transmission == 0.0:
        result["text"] = "No ion reaches the target."
    return result
