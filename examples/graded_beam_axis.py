"""Graded beam axis.

An extraction gap and a long drift on a mesh whose cells grow along the beam.

The planar proton extraction of `beamtracer.examples.beam_optics_extraction_2d` (a plasma
grid at 0 V, an extraction grid at -80 V, 1 mA per metre of depth at 33 eV) followed by a
drift to 120 mm. The beam's optics are set in the gap between the grids, where the
potential changes over a few cells and the electrodes' knife edges must be resolved; along
the drift it hardly changes. A uniform mesh pays for the gap's spacing over the whole
length. `GridParams(x_spacing=...)` grades the beam axis instead: 0.25 mm through the gap,
growing smoothly to 2 mm along the drift, while the spacing across the beam stays 0.25 mm.
`x_through` puts nodes on the faces and tips of the electrodes, so their surfaces fall on
the mesh rather than cutting cells at different places on different meshes. Grading and
refinement boxes combine: a coarser graded mesh (0.5 mm through the gap and across the
beam, 4 mm along the drift) with level-1 boxes over the gap and along the beam has
0.25 mm cells where the beam is, across it as well, and coarse cells elsewhere.

Every mesh solves the same self-consistent beam (the Vlasov iteration with space charge),
and the rms angle and width of the beam at a plane 115 mm downstream are compared with a
uniform mesh of 0.125 mm. The graded mesh, 15,471 nodes, gives the beam of the uniform
0.25 mm mesh, 38,961 nodes: the angle within 0.38 mrad of the reference's 13.68 mrad
against 0.42, the width within 3 %. The uniform 0.5 mm mesh, 9,881 nodes, misses the angle
by 1.21 mrad, three times as much. The graded mesh with the two boxes, 11,404 nodes, is as
close to the reference as the 0.25 mm meshes or closer (+0.08 mrad, the width within 1 %).
A box over the gap alone does little: the drift needs fine cells across the beam, not
along it. The first plot is the cell size along the beam axis, the second the potential
and the trajectories with the graded mesh's lines, the third the beam at the plane against
the number of nodes.

The meshes run one after the other, about 25 s in all.
"""

import time

import numpy as np
import plasma_plots as pp

from beamtracer import BeamOptics, GridParams
from beamtracer.config import RefinementParams
from beamtracer.examples.beam_optics_extraction_2d import create_config, object_outlines

#: The length of the domain along the beam [m] and its height [m].
LENGTH, HEIGHT = 0.120, 0.020
#: The diagnostic plane [m].
PLANE = 0.115
#: The faces and tips of the two grids along x [m]: nodes of the graded mesh.
ELECTRODE_FACES = (0.014, 0.016, 0.018, 0.030, 0.033, 0.036)
#: The node spacing wanted along the beam axis, ``(x, dx)`` [m]: 0.5 mm at the
#: emitter, 0.25 mm through the gap, 2 mm along the drift.
SPACING = [(0.0, 0.5e-3), (0.012, 0.25e-3), (0.037, 0.25e-3), (0.060, 2.0e-3)]
#: The coarser graded mesh under the boxes: 1 mm at the emitter, 0.5 mm through the gap,
#: 4 mm along the drift.
COARSE_SPACING = [(0.0, 1.0e-3), (0.012, 0.5e-3), (0.037, 0.5e-3), (0.060, 4.0e-3)]
#: Level-1 boxes ``(x, y)`` [m]: over the gap, the whole height, and along the beam
#: (y from 8.4 to 11.6 mm at most) to the end of the drift.
BOXES = [
    RefinementParams(level=1, lower=(0.012, 0.0), upper=(0.038, HEIGHT)),
    RefinementParams(level=1, lower=(0.039, 0.007), upper=(LENGTH, 0.013)),
]

# the theme's data colours and ink; the electrodes in grey
ORANGE, AQUA, BLUE, INK, ELECTRODE = "C1", "C2", "C0", "k", "0.7"


def uniform(spacing: float) -> GridParams:
    """A uniform grid of ``spacing`` [m] on both axes."""
    return GridParams(
        num_gridpoints=(round(LENGTH / spacing) + 1, round(HEIGHT / spacing) + 1),
        upper_bound=(LENGTH, HEIGHT),
    )


def graded(spacing=SPACING, across: float = 0.25e-3) -> GridParams:
    """A graded grid: ``spacing`` along the beam, ``across`` [m] across it."""
    return GridParams(
        num_gridpoints=(2, round(HEIGHT / across) + 1),
        upper_bound=(LENGTH, HEIGHT),
        x_spacing=spacing,
        x_through=ELECTRODE_FACES,
    )


def solve(grid: GridParams, boxes=()) -> dict:
    """The self-consistent beam on ``grid`` with the refinement ``boxes`` at the plane."""
    config = create_config(
        grid=grid, refinement=list(boxes), diagnostic_planes=[PLANE], num_particles=400
    )
    start = time.perf_counter()
    model = BeamOptics(config).run()
    seconds = time.perf_counter() - start
    plane = model.state.trace.planes[0]
    forward = plane.direction > 0
    angle = plane.velocity[1][forward] / plane.velocity[0][forward]
    return {
        "model": model,
        "nodes": int(model.state.layout.size),
        "seconds": seconds,
        "angle": 1e3 * float(np.std(angle)),  # mrad
        "width": 1e3 * float(np.std(plane.point[1][forward])),  # mm
    }


def main() -> None:
    """Solve the beam on the meshes and draw the comparison."""
    meshes = {
        "uniform 0.125 mm": uniform(0.125e-3),
        "uniform 0.25 mm": uniform(0.25e-3),
        "uniform 0.5 mm": uniform(0.5e-3),
        "uniform 1 mm": uniform(1.0e-3),
        "graded": graded(),
    }
    results = {name: solve(grid) for name, grid in meshes.items()}
    results["graded + boxes"] = solve(graded(COARSE_SPACING, 0.5e-3), BOXES)
    reference = results["uniform 0.125 mm"]
    print("mesh               nodes   time   rms angle   error   rms width")
    for name, result in results.items():
        print(
            f"{name:17s} {result['nodes']:6d} {result['seconds']:5.1f} s "
            f"{result['angle']:8.2f} mrad {result['angle'] - reference['angle']:+6.2f} "
            f"{result['width']:8.3f} mm"
        )

    # The cell size along the beam axis.
    nodes = meshes["graded"].x_nodes
    with pp.figure(
        figsize=(8, 3.2), title="Cell size along the beam axis of the graded mesh"
    ) as fig:
        ax = fig[0]
        ax.step(nodes[:-1] * 1e3, np.diff(nodes) * 1e3, where="post", color=ORANGE,
                label=f"graded: {nodes.size - 1} cells")  # fmt: skip
        ax.plot(*np.array(SPACING).T * 1e3, "o", color=INK, label="x_spacing")
        for x in ELECTRODE_FACES:
            ax.axvline(x * 1e3, color=ELECTRODE, lw=0.8)
        ax.axhline(0.25, color=AQUA, ls="--", lw=1, label="uniform 0.25 mm")
        ax.set_ylim(0.0, 2.2)
        ax.set_xlabel("$x$ [mm]  (grey: the electrodes' faces)")
        ax.set_ylabel("cell size [mm]")
        ax.legend(loc="upper left")

    # The potential and the trajectories, with the mesh lines along the beam.
    model = results["graded"]["model"]
    potential = model.potential_field.evaluate()
    with pp.figure(
        figsize=(9, 3.4),
        title="Potential and trajectories on the graded mesh (every 4th line along x)",
    ) as fig:
        ax = fig[0]
        mesh = ax.pcolormesh(potential.x * 1e3, potential.y * 1e3, potential.T,
                             cmap="Blues_r", shading="auto")  # fmt: skip
        ax.figure.colorbar(mesh, ax=ax, label="potential [V]", pad=0.01)
        for x in nodes[::4]:
            ax.axvline(x * 1e3, color=INK, lw=0.3, alpha=0.4)
        for x, y in object_outlines(model.config).values():
            ax.fill(x * 1e3, y * 1e3, color=ELECTRODE, lw=0)
        for trajectory in model.trajectories:
            ax.plot(
                trajectory[:, 0] * 1e3, trajectory[:, 1] * 1e3, color=ORANGE, lw=0.7
            )
        ax.axvline(PLANE * 1e3, color=INK, ls="--", lw=1)
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")

    # The beam at the plane against the number of nodes.
    with pp.figure(
        1, 2, figsize=(9, 3.4), title=f"The beam at x = {PLANE * 1e3:.0f} mm"
    ) as fig:
        for ax, key, label in (
            (fig[0], "angle", "rms angle [mrad]"),
            (fig[1], "width", "rms width [mm]"),
        ):
            names = [name for name in results if name.startswith("uniform")]
            ax.plot([results[n]["nodes"] for n in names],
                    [results[n][key] for n in names], "o-", color=BLUE,
                    label="uniform 1, 0.5, 0.25, 0.125 mm")  # fmt: skip
            ax.plot(results["graded"]["nodes"], results["graded"][key], "s",
                    color=ORANGE, markersize=8, label="graded")  # fmt: skip
            ax.plot(results["graded + boxes"]["nodes"], results["graded + boxes"][key],
                    "D", color="C3", markersize=8, label="graded + boxes")  # fmt: skip
            ax.axhline(reference[key], color=INK, ls="--", lw=1, label="0.125 mm")
            ax.set_xscale("log")
            ax.set_xlabel("nodes")
            ax.set_ylabel(label)
        fig[0].legend(loc="upper right")


if __name__ == "__main__":
    main()
