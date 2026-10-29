"""The animated figure at the top of the docs home page: ``docs/src/assets/hero.svg``.

A proton beamlet extracted from a plasma through a plasma grid and an extraction grid: the
self-consistent trajectories of :mod:`beamtracer.examples.beam_optics_extraction_2d` (a
:class:`~beamtracer.BeamOptics` Vlasov iteration, with fewer particles so that it takes a couple
of seconds) over the equipotentials. Ions run along the trajectories;
those that hit the grids stop there. Rerun with ``make docs-hero`` or ``python docs/hero.py``.
"""

from pathlib import Path

import contourpy
import numpy as np
from plasma_plots.hero import HeroFigure

from beamtracer.examples.beam_optics_extraction_2d import create_config, object_outlines
from beamtracer.model import BeamOptics

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
MM = 1e3

config = create_config(num_particles=48, save_trajectories=1)
model = BeamOptics(config).run()

# the whole 20 mm high domain, and as much room upstream as the 8:3 picture leaves
fig = HeroFigure(xlim=(40.0 - 20 * 8 / 3, 40.0), ylim=(0.0, 20.0))

potential = model.potential_field.evaluate(name="potential", units="V")
x, y, phi = potential.x.values * MM, potential.y.values * MM, potential.values.T
generator = contourpy.contour_generator(x, y, phi)
for level in np.linspace(-90.0, -5.0, 12):
    for line in generator.lines(level):
        fig.line(line[::2, 0], line[::2, 1], tone="muted", width=1.0, opacity=0.4)

for name, (ox, oy) in object_outlines(config).items():
    breaks = np.flatnonzero(np.isnan(ox))
    for px, py in zip(np.split(ox, breaks), np.split(oy, breaks), strict=True):
        keep = np.isfinite(px)
        fig.polygon(
            px[keep] * MM, py[keep] * MM, tone="low", outline="accent", title=name
        )

paths = [t * MM for t in model.trajectories]
for i, path in enumerate(paths):
    px, py = np.r_[path[::2, 0], path[-1, 0]], np.r_[path[::2, 1], path[-1, 1]]
    transmitted = path[-1, 0] > 39.0
    if not transmitted and i % 3:
        continue  # a few of the ions that end on the plasma grid
    fig.line(px, py, tone="accent", width=1.0, opacity=0.45 if transmitted else 0.2)
    duration = 8.0 if transmitted else 4.0
    fig.tracer(px, y=py, duration=duration, offset=duration * i / len(paths), r=3.0)

fig.save(OUTPUT)
print(f"{OUTPUT}: {OUTPUT.stat().st_size / 1e3:.0f} kB")
