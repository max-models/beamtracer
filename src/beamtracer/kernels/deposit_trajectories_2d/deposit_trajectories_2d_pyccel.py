"""The Pyccel version (CPU) of the ``deposit_trajectories_2d`` kernel of the ion-optics tracer.

``deposit_trajectories_2d_cuda.cu`` next to this file is the CUDA version (GPU) and
``deposit_trajectories_2d_numpy.py`` the version for machines without Pyccel, all with the
same arguments; :class:`beamtracer.solver.BeamTracer` calls it.
"""

# mypy: ignore-errors

from math import floor


def deposit_trajectories_2d(
    px: "float[:]",
    py: "float[:]",
    q: "float[:]",
    origin_x: float,
    origin_y: float,
    h: float,
    cylindrical: int,
    charge: "float[:,:]",
) -> None:
    """Add the charges ``q`` at ``(px, py)`` to the node charges, area-weighted.

    Bilinear in ``x``; in ``y`` linear, or in cylindrical geometry (``y`` the radius)
    linear in ``r**2``, so a uniform density gives node charges proportional to the
    ring volumes. Charge beyond the mesh goes to its boundary nodes.
    """
    nx = charge.shape[0]
    ny = charge.shape[1]
    for k in range(px.shape[0]):
        x = px[k]
        y = py[k]
        i = floor((x - origin_x) / h)
        j = floor((y - origin_y) / h)
        t = (x - (origin_x + h * i)) / h
        if cylindrical != 0:
            r1 = origin_y + h * j
            r2 = r1 + h
            u = (y * y - r1 * r1) / (r2 * r2 - r1 * r1)
        else:
            u = (y - (origin_y + h * j)) / h
        if i < 0:
            t = 0.0
            i = 0
        elif i >= nx - 1:
            t = 1.0
            i = nx - 2
        if j < 0:
            u = 0.0
            j = 0
        elif j >= ny - 1:
            u = 1.0
            j = ny - 2
        value = q[k]
        charge[i, j] += (1.0 - t) * (1.0 - u) * value
        charge[i, j + 1] += (1.0 - t) * u * value
        charge[i + 1, j] += t * (1.0 - u) * value
        charge[i + 1, j + 1] += t * u * value
