"""Relativistic electrons.

Electrons accelerated to 1 MeV, and their gyration in a magnetic field, traced classically
and with ``relativistic=True``.

At 1 MeV an electron has ``gamma = 1 + E / (m c^2) = 2.96`` and moves at 0.94 c; the
classical speed of that energy, ``sqrt(2 E / m)``, would be twice the speed of light.
`BeamTracer(relativistic=True)` (``relativistic: true`` in a `BeamOpticsConfig`) advances
the proper velocity ``u = gamma v`` instead of the velocity: the electric field changes
``u`` at the rate ``q E / m``, the positions move with ``u / gamma``, and the magnetic
force is ``q u x B / gamma`` (Boris's step is mpiparticles' relativistic one). The plane
crossings record the velocities ``u / gamma``.

The first plot follows electrons from 1 km/s through a uniform field of 50 MV/m, 1 MV over
2 cm: the relativistic speed at 40 diagnostic planes against the exact hyperbolic motion,
``gamma(x) = gamma_0 + q E x / (m c^2)``, and the classical one, which passes ``c`` after a
quarter of the gap. The second plot is a 1 MeV electron in 0.1 T: it gyrates on the radius
``gamma m v / (q B) = 47 mm``, three times the classical 16 mm of an electron traced
classically at the same speed (0.94 c), which also turns three times as fast.
"""

import numpy as np
import plasma_plots as pp
from scipy import constants

from beamtracer.solver import (
    BeamParticles,
    BeamTracer,
    IonOpticsMesh,
    StaggeredElectricField,
    beam_with_energy,
)

C, E, M_E = constants.c, constants.e, constants.m_e
#: The gap [m] and the voltage across it [V].
LENGTH, VOLTAGE = 0.02, 1.0e6
#: The magnetic field of the gyration [T] and the electrons' energy there [eV].
FIELD, ENERGY = 0.1, 1.0e6
# the theme's data colours and ink
ORANGE, AQUA, INK = "C1", "C2", "k"


def electrons(position: np.ndarray, velocity: np.ndarray) -> BeamParticles:
    """Electrons at ``position`` with ``velocity`` ``(3, N)``, 1 uA each."""
    n = position.shape[1]
    return BeamParticles(
        position, velocity, np.full(n, -E), np.full(n, M_E), np.full(n, 1e-6),
        np.zeros(n, dtype=np.int32),
    )  # fmt: skip


def acceleration() -> dict:
    """The speed of an electron at 40 planes through the gap, both ways, and the exact one."""
    mesh = IonOpticsMesh((81, 5), (0.0, 0.0), LENGTH / 80, "planar")
    x = mesh.position_x(np.arange(81))
    field = StaggeredElectricField(mesh, VOLTAGE / LENGTH * x[:, None] * np.ones(5))
    planes = np.linspace(0.0005, LENGTH - 0.0005, 40)
    start = electrons(np.array([[0.0], [1e-4], [0.0]]), np.array([[1e3], [0.0], [0.0]]))
    speeds = {}
    for relativistic in (False, True):
        tracer = BeamTracer(
            mesh, field, planes=planes, relativistic=relativistic, step_fraction=0.25
        )
        result = tracer.trace(start)
        speeds[relativistic] = [float(hits.velocity[0, 0]) for hits in result.planes]
    gamma = 1.0 + E * VOLTAGE / LENGTH * planes / (M_E * C**2)
    return {
        "x": planes,
        "classical": np.array(speeds[False]),
        "relativistic": np.array(speeds[True]),
        "exact": C * np.sqrt(1.0 - 1.0 / gamma**2),
    }


def true_speed(velocity: np.ndarray) -> np.ndarray:
    """The velocity of a 1 MeV electron along ``velocity``'s direction: 0.94 c."""
    gamma = 1.0 + ENERGY * E / (M_E * C**2)
    direction = velocity / np.linalg.norm(velocity, axis=0)
    return direction * C * np.sqrt(1.0 - 1.0 / gamma**2)


def gyration() -> dict:
    """The orbit of a 1 MeV electron in ``FIELD`` along z, both ways, about the origin."""
    mesh = IonOpticsMesh((241, 241), (-0.06, -0.06), 0.5e-3, "planar")
    field = StaggeredElectricField(mesh, np.zeros(mesh.shape))

    def magnetic(x, y):
        return np.zeros_like(x), np.zeros_like(x), np.full_like(x, FIELD)

    mass = M_E / constants.atomic_mass  # [u]
    orbits = {}
    for relativistic in (False, True):
        beam, _ = beam_with_energy(
            1, 1.0, -1.0, mass, ENERGY, 0.0, 0.0, (0.0, -0.048), (0.0, -0.046),
            cylindrical=False, relativistic=relativistic,
        )  # fmt: skip
        if not relativistic:  # the classical tracer with the true speed, 0.94 c
            beam.velocity = true_speed(beam.velocity)
        tracer = BeamTracer(
            mesh, field, magnetic, integrator="boris", relativistic=relativistic,
            save_trajectories=1, step_fraction=0.2, max_steps=4000,
        )  # fmt: skip
        orbits[relativistic] = tracer.trace(beam).trajectories[0]
    return orbits


def main() -> None:
    """Trace both cases and draw them."""
    gap = acceleration()
    error = np.abs(gap["relativistic"] / gap["exact"] - 1.0).max()
    print(f"relativistic speed at the end {gap['relativistic'][-1] / C:.4f} c, "
          f"largest error {error:.1e}; classical {gap['classical'][-1] / C:.2f} c")  # fmt: skip
    with pp.figure(figsize=(8, 3.2), title="An electron through 1 MV over 2 cm") as fig:
        ax = fig[0]
        ax.plot(gap["x"] * 1e3, gap["classical"] / C, "o-", color=ORANGE, ms=3,
                label="classical")  # fmt: skip
        ax.plot(gap["x"] * 1e3, gap["relativistic"] / C, "o", color=AQUA, ms=4,
                label="relativistic")  # fmt: skip
        ax.plot(gap["x"] * 1e3, gap["exact"] / C, color=INK, lw=1, label="exact")
        ax.axhline(1.0, color=INK, ls="--", lw=0.8)
        ax.set_ylim(0.0, 2.0)
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$v / c$")
        ax.legend(loc="lower right")

    orbits = gyration()
    gamma = 1.0 + ENERGY * E / (M_E * C**2)
    radius = gamma * M_E * C * np.sqrt(1.0 - 1.0 / gamma**2) / (E * FIELD)
    print(f"gyroradius: relativistic {radius * 1e3:.1f} mm, classical "
          f"{radius / gamma * 1e3:.1f} mm")  # fmt: skip
    with pp.figure(figsize=(5, 5), title="A 1 MeV electron in 0.1 T") as fig:
        ax = fig[0]
        ax.plot(*orbits[True].T * 1e3, color=AQUA, label="relativistic")
        ax.plot(*orbits[False].T * 1e3, color=ORANGE, label="classical")
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [mm]")
        ax.set_ylabel("$y$ [mm]")
        ax.legend(loc="upper right")


if __name__ == "__main__":
    main()
