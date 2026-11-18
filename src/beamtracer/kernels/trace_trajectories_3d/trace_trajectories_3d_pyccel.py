"""The Pyccel version (CPU) of the ``trace_trajectories_3d`` kernel of the 3D tracer.

``trace_trajectories_3d_cuda.cu`` next to this file is the CUDA version (GPU), with the
same arguments; `beamtracer.solver.BeamTracer3D` calls it. Without Pyccel cunumpy runs this
source as Python.

As in ``trace_trajectories_2d``, the field gathers and the deposit are written out in the
loop rather than called: Pyccel passes arrays to a function as Fortran descriptors, rebuilt
at every call, which costs more than the interpolation itself.
"""

# mypy: ignore-errors

from math import floor, sqrt


def trace_trajectories_3d(
    active: "int[:]",
    x: "float[:,:]",
    v: "float[:,:]",
    qm: "float[:]",
    dq: "float[:]",
    steps: "int[:]",
    status: "int[:]",
    electrode: "int[:]",
    fx: "float[:,:,:]",
    fy: "float[:,:,:]",
    fz: "float[:,:,:]",
    bn: "float[:,:,:,:]",
    has_b: int,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    h: float,
    upper_x: float,
    upper_y: float,
    upper_z: float,
    mirror_y: int,
    mirror_z: int,
    integrator: int,
    step_fraction: float,
    max_steps: int,
    cells: "int[:,:,:]",
    subcells: "int[:]",
    refine: int,
    planes: "float[:]",
    hits: "float[:,:,:,:]",
    counts: "int[:,:]",
    path_slot: "int[:]",
    paths: "float[:,:,:]",
    path_count: "int[:]",
    num_threads: int,
    charge: "float[:,:]",
) -> None:
    """Trace the ``active`` trajectories until they end, or reach a surface.

    ``x``, ``v`` ``(3, N)`` are the states (updated in place), ``qm`` the charge over mass,
    ``dq`` the signed current deposited per unit time, ``steps`` the steps taken so far.
    The electric field is the edge fields ``fx`` ``(nx + 1, ny, nz)``, ``fy``
    ``(nx, ny + 1, nz)`` and ``fz`` ``(nx, ny, nz + 1)`` (as `StaggeredElectricField3D`),
    gathered trilinearly; with ``has_b`` the magnetic field ``bn`` ``(3, nx, ny, nz)`` on
    the nodes, gathered trilinearly too. A step is the Runge--Kutta step (``integrator``
    0) of ``dv/dt = qm (E + v x B)`` or Boris's kick--rotate--kick in two half steps of
    ``dt / 2`` around the drift (1, one field evaluation a step: the end's is the next
    start's). The step is ``step_fraction`` of ``min(h / |v|, sqrt(2 h / |a|))``. It
    deposits ``dq dt`` at its Hermite midpoint into ``charge[t]`` (the nodes flattened,
    trilinear weights), one slab per thread ``t < num_threads``, and records every crossing
    of the ``planes`` (``x = const``) in ``hits[k, plane, c]`` (point, velocity, direction,
    step) for the first ``hits.shape[2]`` crossings, counting all in ``counts[k, plane]``.

    With ``mirror_y`` (``mirror_z``) the lower ``y`` (``z``) face is a symmetry plane: a
    point below it sees the field of its mirror image, ``E_y`` (``E_z``) and the other two
    components of ``B`` changing sign, deposits there, and only the upper face is an exit.

    ``status`` ends as 0-4 (transmitted, backward, wall, electrode, max steps), or -2 for an
    end point in a sub-cell that an electrode surface cuts: the caller decides those with
    the electrodes' own test and resumes the others. ``cells`` gives each cell its block of
    ``refine**3`` sub-cells (-1: no electrode near), whose entry is the electrode
    containing the sub-cell, -1 for none or -2. A trajectory with ``path_slot[k] >= 0``
    appends the mesh coordinates of every step's end to ``paths[slot]``, as far as it
    holds, counting all in ``path_count[slot]``. Thread ``t`` traces every
    ``num_threads``-th active trajectory.
    """
    nx = fy.shape[0]
    ny = fx.shape[1]
    nz = fx.shape[2]
    n_active = active.shape[0]
    n_planes = planes.shape[0]
    capacity = hits.shape[2]
    path_capacity = paths.shape[1]
    # fmt: off
    #$ omp parallel for schedule(static, 1) num_threads(num_threads) private(t, m, k, slot, nstage, have, ce0, ce1, ce2, cb0, cb1, cb2, x0, x1, x2, v0, v1, v2, qmk, ns, st, stage, q0, q1, q2, p0, p1, p2, s1, s2, sx, sy, sz, s, i, j, kz, a, b, c, w000, w001, w010, w011, w100, w101, w110, w111, n0, e0, e1, e2, b0, b1, b2, g0, g1, g2, a0, a1, a2, kx0, kx1, kx2, sx0, sx1, sx2, sv0, sv1, sv2, speed, accel, dt, d2, hdt, dt6, y0, y1, y2, w0, w1, w2, m0, m1, m2, deposit, p, xp, before, after, cc, px, py, pz, cell, si, sj, sk, e, tau, t0, t1, t2, tt, f, u0, u1, u2, r0, r1, r2)
    # fmt: on
    for t in range(num_threads):
        m = t
        while m < n_active:
            k = active[m]
            m = m + num_threads
            x0 = x[0, k]
            x1 = x[1, k]
            x2 = x[2, k]
            v0 = v[0, k]
            v1 = v[1, k]
            v2 = v[2, k]
            qmk = qm[k]
            slot = path_slot[k]
            ns = steps[k]
            st = -1
            dt = 0.0
            hdt = 0.0
            dt6 = 0.0
            tau = 0.0
            nstage = 4 if integrator == 0 else 2
            have = 0
            ce0 = 0.0
            ce1 = 0.0
            ce2 = 0.0
            cb0 = 0.0
            cb1 = 0.0
            cb2 = 0.0
            while ns < max_steps:
                q0 = x0
                q1 = x1
                q2 = x2
                kx0 = v0
                kx1 = v1
                kx2 = v2
                for stage in range(nstage):
                    # the fields at q (mirrored below a symmetry plane), or for Boris at the
                    # start those of the last step's end
                    if stage == 0 and have != 0:
                        e0 = ce0
                        e1 = ce1
                        e2 = ce2
                        b0 = cb0
                        b1 = cb1
                        b2 = cb2
                    else:
                        p0 = q0
                        p1 = q1
                        p2 = q2
                        s1 = 1.0
                        s2 = 1.0
                        if mirror_y != 0 and p1 < origin_y:
                            p1 = 2.0 * origin_y - p1
                            s1 = -1.0
                        if mirror_z != 0 and p2 < origin_z:
                            p2 = 2.0 * origin_z - p2
                            s2 = -1.0
                        sx = (p0 - origin_x) / h
                        sy = (p1 - origin_y) / h
                        sz = (p2 - origin_z) / h
                        # E_x on (nx + 1, ny, nz) at (sx + 1/2, sy, sz)
                        s = sx + 0.5
                        i = floor(s)
                        if i < 0:
                            i = 0
                        elif i > nx - 1:
                            i = nx - 1
                        j = floor(sy)
                        if j < 0:
                            j = 0
                        elif j > ny - 2:
                            j = ny - 2
                        kz = floor(sz)
                        if kz < 0:
                            kz = 0
                        elif kz > nz - 2:
                            kz = nz - 2
                        a = s - i
                        b = sy - j
                        c = sz - kz
                        e0 = (
                            (1.0 - a) * ((1.0 - b) * ((1.0 - c) * fx[i, j, kz] + c * fx[i, j, kz + 1])
                                         + b * ((1.0 - c) * fx[i, j + 1, kz] + c * fx[i, j + 1, kz + 1]))
                            + a * ((1.0 - b) * ((1.0 - c) * fx[i + 1, j, kz] + c * fx[i + 1, j, kz + 1])
                                   + b * ((1.0 - c) * fx[i + 1, j + 1, kz] + c * fx[i + 1, j + 1, kz + 1]))
                        )  # fmt: skip
                        # E_y on (nx, ny + 1, nz) at (sx, sy + 1/2, sz)
                        i = floor(sx)
                        if i < 0:
                            i = 0
                        elif i > nx - 2:
                            i = nx - 2
                        s = sy + 0.5
                        j = floor(s)
                        if j < 0:
                            j = 0
                        elif j > ny - 1:
                            j = ny - 1
                        a = sx - i
                        b = s - j
                        e1 = s1 * (
                            (1.0 - a) * ((1.0 - b) * ((1.0 - c) * fy[i, j, kz] + c * fy[i, j, kz + 1])
                                         + b * ((1.0 - c) * fy[i, j + 1, kz] + c * fy[i, j + 1, kz + 1]))
                            + a * ((1.0 - b) * ((1.0 - c) * fy[i + 1, j, kz] + c * fy[i + 1, j, kz + 1])
                                   + b * ((1.0 - c) * fy[i + 1, j + 1, kz] + c * fy[i + 1, j + 1, kz + 1]))
                        )  # fmt: skip
                        # E_z on (nx, ny, nz + 1) at (sx, sy, sz + 1/2)
                        j = floor(sy)
                        if j < 0:
                            j = 0
                        elif j > ny - 2:
                            j = ny - 2
                        s = sz + 0.5
                        kz = floor(s)
                        if kz < 0:
                            kz = 0
                        elif kz > nz - 1:
                            kz = nz - 1
                        b = sy - j
                        c = s - kz
                        e2 = s2 * (
                            (1.0 - a) * ((1.0 - b) * ((1.0 - c) * fz[i, j, kz] + c * fz[i, j, kz + 1])
                                         + b * ((1.0 - c) * fz[i, j + 1, kz] + c * fz[i, j + 1, kz + 1]))
                            + a * ((1.0 - b) * ((1.0 - c) * fz[i + 1, j, kz] + c * fz[i + 1, j, kz + 1])
                                   + b * ((1.0 - c) * fz[i + 1, j + 1, kz] + c * fz[i + 1, j + 1, kz + 1]))
                        )  # fmt: skip
                        b0 = 0.0
                        b1 = 0.0
                        b2 = 0.0
                        if has_b != 0:
                            # B on the nodes (3, nx, ny, nz) at (sx, sy, sz)
                            kz = floor(sz)
                            if kz < 0:
                                kz = 0
                            elif kz > nz - 2:
                                kz = nz - 2
                            c = sz - kz
                            w000 = (1.0 - a) * (1.0 - b) * (1.0 - c)
                            w001 = (1.0 - a) * (1.0 - b) * c
                            w010 = (1.0 - a) * b * (1.0 - c)
                            w011 = (1.0 - a) * b * c
                            w100 = a * (1.0 - b) * (1.0 - c)
                            w101 = a * (1.0 - b) * c
                            w110 = a * b * (1.0 - c)
                            w111 = a * b * c
                            for n0 in range(3):
                                f = (
                                    w000 * bn[n0, i, j, kz] + w001 * bn[n0, i, j, kz + 1]
                                    + w010 * bn[n0, i, j + 1, kz] + w011 * bn[n0, i, j + 1, kz + 1]
                                    + w100 * bn[n0, i + 1, j, kz] + w101 * bn[n0, i + 1, j, kz + 1]
                                    + w110 * bn[n0, i + 1, j + 1, kz] + w111 * bn[n0, i + 1, j + 1, kz + 1]
                                )  # fmt: skip
                                if n0 == 0:
                                    b0 = s1 * s2 * f
                                elif n0 == 1:
                                    b1 = s2 * f
                                else:
                                    b2 = s1 * f
                    # the acceleration with the stage's velocity kx
                    g0 = qmk * (e0 + kx1 * b2 - kx2 * b1)
                    g1 = qmk * (e1 + kx2 * b0 - kx0 * b2)
                    g2 = qmk * (e2 + kx0 * b1 - kx1 * b0)
                    if stage == 0:
                        a0 = g0
                        a1 = g1
                        a2 = g2
                        speed = sqrt(v0 * v0 + v1 * v1 + v2 * v2)
                        accel = sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                        dt = -1.0
                        if speed > 0.0:
                            dt = h / speed
                        if accel > 0.0:
                            d2 = sqrt(2.0 * h / accel)
                            if dt < 0.0 or d2 < dt:
                                dt = d2
                        if dt < 0.0:
                            dt = h
                        dt = step_fraction * dt
                        hdt = 0.5 * dt
                        dt6 = dt / 6.0
                        if integrator != 0:
                            # Boris: a half step's kick--rotate--kick, then the drift
                            tau = 0.5 * hdt * qmk
                            u0 = v0 + tau * e0
                            u1 = v1 + tau * e1
                            u2 = v2 + tau * e2
                            t0 = tau * b0
                            t1 = tau * b1
                            t2 = tau * b2
                            tt = 2.0 / (1.0 + t0 * t0 + t1 * t1 + t2 * t2)
                            r0 = u0 + u1 * t2 - u2 * t1
                            r1 = u1 + u2 * t0 - u0 * t2
                            r2 = u2 + u0 * t1 - u1 * t0
                            u0 = u0 + tt * (r1 * t2 - r2 * t1)
                            u1 = u1 + tt * (r2 * t0 - r0 * t2)
                            u2 = u2 + tt * (r0 * t1 - r1 * t0)
                            kx0 = u0 + tau * e0
                            kx1 = u1 + tau * e1
                            kx2 = u2 + tau * e2
                            q0 = x0 + dt * kx0
                            q1 = x1 + dt * kx1
                            q2 = x2 + dt * kx2
                            continue
                        sx0 = v0
                        sx1 = v1
                        sx2 = v2
                        sv0 = a0
                        sv1 = a1
                        sv2 = a2
                        q0 = x0 + hdt * v0
                        q1 = x1 + hdt * v1
                        q2 = x2 + hdt * v2
                        kx0 = v0 + hdt * a0
                        kx1 = v1 + hdt * a1
                        kx2 = v2 + hdt * a2
                    elif integrator != 0:
                        # Boris: the second half step's kick--rotate--kick at the end
                        u0 = kx0 + tau * e0
                        u1 = kx1 + tau * e1
                        u2 = kx2 + tau * e2
                        t0 = tau * b0
                        t1 = tau * b1
                        t2 = tau * b2
                        tt = 2.0 / (1.0 + t0 * t0 + t1 * t1 + t2 * t2)
                        r0 = u0 + u1 * t2 - u2 * t1
                        r1 = u1 + u2 * t0 - u0 * t2
                        r2 = u2 + u0 * t1 - u1 * t0
                        u0 = u0 + tt * (r1 * t2 - r2 * t1)
                        u1 = u1 + tt * (r2 * t0 - r0 * t2)
                        u2 = u2 + tt * (r0 * t1 - r1 * t0)
                        w0 = u0 + tau * e0
                        w1 = u1 + tau * e1
                        w2 = u2 + tau * e2
                        ce0 = e0
                        ce1 = e1
                        ce2 = e2
                        cb0 = b0
                        cb1 = b1
                        cb2 = b2
                        have = 1
                    elif stage == 1:
                        sx0 = sx0 + 2.0 * kx0
                        sx1 = sx1 + 2.0 * kx1
                        sx2 = sx2 + 2.0 * kx2
                        sv0 = sv0 + 2.0 * g0
                        sv1 = sv1 + 2.0 * g1
                        sv2 = sv2 + 2.0 * g2
                        q0 = x0 + hdt * kx0
                        q1 = x1 + hdt * kx1
                        q2 = x2 + hdt * kx2
                        kx0 = v0 + hdt * g0
                        kx1 = v1 + hdt * g1
                        kx2 = v2 + hdt * g2
                    elif stage == 2:
                        sx0 = sx0 + 2.0 * kx0
                        sx1 = sx1 + 2.0 * kx1
                        sx2 = sx2 + 2.0 * kx2
                        sv0 = sv0 + 2.0 * g0
                        sv1 = sv1 + 2.0 * g1
                        sv2 = sv2 + 2.0 * g2
                        q0 = x0 + dt * kx0
                        q1 = x1 + dt * kx1
                        q2 = x2 + dt * kx2
                        kx0 = v0 + dt * g0
                        kx1 = v1 + dt * g1
                        kx2 = v2 + dt * g2
                    else:
                        sx0 = sx0 + kx0
                        sx1 = sx1 + kx1
                        sx2 = sx2 + kx2
                        sv0 = sv0 + g0
                        sv1 = sv1 + g1
                        sv2 = sv2 + g2
                if integrator == 0:
                    y0 = x0 + dt6 * sx0
                    y1 = x1 + dt6 * sx1
                    y2 = x2 + dt6 * sx2
                    w0 = v0 + dt6 * sv0
                    w1 = v1 + dt6 * sv1
                    w2 = v2 + dt6 * sv2
                else:
                    y0 = q0
                    y1 = q1
                    y2 = q2

                # space charge: dq dt at the Hermite midpoint, trilinear
                m0 = 0.5 * (x0 + y0) + dt * (v0 - w0) / 8.0
                m1 = 0.5 * (x1 + y1) + dt * (v1 - w1) / 8.0
                m2 = 0.5 * (x2 + y2) + dt * (v2 - w2) / 8.0
                if mirror_y != 0 and m1 < origin_y:
                    m1 = 2.0 * origin_y - m1
                if mirror_z != 0 and m2 < origin_z:
                    m2 = 2.0 * origin_z - m2
                sx = (m0 - origin_x) / h
                sy = (m1 - origin_y) / h
                sz = (m2 - origin_z) / h
                i = floor(sx)
                j = floor(sy)
                kz = floor(sz)
                a = sx - i
                b = sy - j
                c = sz - kz
                if i < 0:
                    a = 0.0
                    i = 0
                elif i >= nx - 1:
                    a = 1.0
                    i = nx - 2
                if j < 0:
                    b = 0.0
                    j = 0
                elif j >= ny - 1:
                    b = 1.0
                    j = ny - 2
                if kz < 0:
                    c = 0.0
                    kz = 0
                elif kz >= nz - 1:
                    c = 1.0
                    kz = nz - 2
                deposit = dq[k] * dt
                cc = (i * ny + j) * nz + kz
                charge[t, cc] += (1.0 - a) * (1.0 - b) * (1.0 - c) * deposit
                charge[t, cc + 1] += (1.0 - a) * (1.0 - b) * c * deposit
                charge[t, cc + nz] += (1.0 - a) * b * (1.0 - c) * deposit
                charge[t, cc + nz + 1] += (1.0 - a) * b * c * deposit
                charge[t, cc + ny * nz] += a * (1.0 - b) * (1.0 - c) * deposit
                charge[t, cc + ny * nz + 1] += a * (1.0 - b) * c * deposit
                charge[t, cc + ny * nz + nz] += a * b * (1.0 - c) * deposit
                charge[t, cc + ny * nz + nz + 1] += a * b * c * deposit
                ns = ns + 1

                for p in range(n_planes):
                    xp = planes[p]
                    before = x0 - xp
                    after = y0 - xp
                    if (before < 0.0) != (after < 0.0):
                        s = before / (before - after)
                        cc = counts[k, p]
                        if cc < capacity:
                            hits[k, p, cc, 0] = xp
                            hits[k, p, cc, 1] = x1 + s * (y1 - x1)
                            hits[k, p, cc, 2] = x2 + s * (y2 - x2)
                            hits[k, p, cc, 3] = v0 + s * (w0 - v0)
                            hits[k, p, cc, 4] = v1 + s * (w1 - v1)
                            hits[k, p, cc, 5] = v2 + s * (w2 - v2)
                            hits[k, p, cc, 6] = 1.0 if after >= 0.0 else -1.0
                            hits[k, p, cc, 7] = 1.0 * ns
                        counts[k, p] = cc + 1
                x0 = y0
                x1 = y1
                x2 = y2
                v0 = w0
                v1 = w1
                v2 = w2

                # the end of the step: off the mesh or in an electrode
                px = x0
                py = x1
                pz = x2
                if mirror_y != 0 and py < origin_y:
                    py = 2.0 * origin_y - py
                if mirror_z != 0 and pz < origin_z:
                    pz = 2.0 * origin_z - pz
                if slot >= 0:
                    cc = path_count[slot]
                    if cc < path_capacity:
                        paths[slot, cc, 0] = px
                        paths[slot, cc, 1] = py
                        paths[slot, cc, 2] = pz
                    path_count[slot] = cc + 1
                if px >= upper_x:
                    st = 0
                elif px < origin_x:
                    st = 1
                elif py > upper_y or py < origin_y or pz > upper_z or pz < origin_z:
                    st = 2
                else:
                    i = int((px - origin_x) / h)
                    if i > nx - 2:
                        i = nx - 2
                    j = int((py - origin_y) / h)
                    if j > ny - 2:
                        j = ny - 2
                    kz = int((pz - origin_z) / h)
                    if kz > nz - 2:
                        kz = nz - 2
                    cell = cells[i, j, kz]
                    if cell >= 0:
                        si = int(((px - origin_x) / h - i) * refine)
                        if si < 0:
                            si = 0
                        elif si > refine - 1:
                            si = refine - 1
                        sj = int(((py - origin_y) / h - j) * refine)
                        if sj < 0:
                            sj = 0
                        elif sj > refine - 1:
                            sj = refine - 1
                        sk = int(((pz - origin_z) / h - kz) * refine)
                        if sk < 0:
                            sk = 0
                        elif sk > refine - 1:
                            sk = refine - 1
                        e = subcells[((cell * refine + si) * refine + sj) * refine + sk]
                        if e >= 0:
                            st = 3
                            electrode[k] = e
                        elif e == -2:
                            st = -2
                if st != -1:
                    break
            if st == -1:
                st = 4
            x[0, k] = x0
            x[1, k] = x1
            x[2, k] = x2
            v[0, k] = v0
            v[1, k] = v1
            v[2, k] = v2
            steps[k] = ns
            status[k] = st
