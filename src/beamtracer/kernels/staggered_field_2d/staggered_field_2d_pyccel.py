"""The Pyccel version (CPU) of the ``staggered_field_2d`` kernel of the ion-optics tracer.

``staggered_field_2d_cuda.cu`` next to this file is the CUDA version (GPU) and
``staggered_field_2d_numpy.py`` the version for machines without Pyccel, all with the
same arguments; :class:`beamtracer.solver.StaggeredElectricField` calls it.
"""

# mypy: ignore-errors

from math import floor


def staggered_field_2d(
    px: "float[:]",
    py: "float[:]",
    fx: "float[:,:]",
    fy: "float[:,:]",
    origin_x: float,
    origin_y: float,
    h: float,
    ex: "float[:]",
    ey: "float[:]",
) -> None:
    """``(E_x, E_y)`` at the points ``(px, py)`` from the edge fields.

    ``fx`` ``(nx + 1, ny)`` holds ``E_x`` on the x edges (between nodes along x) and
    ``fy`` ``(nx, ny + 1)`` ``E_y`` on the y edges; each is interpolated bilinearly,
    with the cell clamped to the array (points just outside are extrapolated).
    """
    nx = fy.shape[0]
    ny = fx.shape[1]
    # fmt: off
    #$ omp parallel for private(sx, sy, s, i, j, u, w, low, high, t) schedule(dynamic, 4096)
    # fmt: on
    for k in range(px.shape[0]):
        sx = (px[k] - origin_x) / h
        sy = (py[k] - origin_y) / h
        # E_x: fractional indices (sx + 1/2, sy) in fx
        s = sx + 0.5
        i = min(max(floor(s), 0), nx - 1)
        j = min(max(floor(sy), 0), ny - 2)
        u = s - i
        w = sy - j
        low = fx[i, j] + w * (fx[i, j + 1] - fx[i, j])
        high = fx[i + 1, j] + w * (fx[i + 1, j + 1] - fx[i + 1, j])
        ex[k] = low + u * (high - low)
        # E_y: fractional indices (sx, sy + 1/2) in fy
        t = sy + 0.5
        i = min(max(floor(sx), 0), nx - 2)
        j = min(max(floor(t), 0), ny - 1)
        u = sx - i
        w = t - j
        low = fy[i, j] + w * (fy[i, j + 1] - fy[i, j])
        high = fy[i + 1, j] + w * (fy[i + 1, j + 1] - fy[i + 1, j])
        ey[k] = low + u * (high - low)
