"""The Pyccel version (CPU) of the ``trace_trajectories_2d`` kernel of the ion-optics tracer.

``trace_trajectories_2d_cuda.cu`` next to this file is the CUDA version (GPU), with the
same arguments; `beamtracer.solver.BeamTracer` calls it. There is no NumPy version: without
Pyccel the tracer runs its NumPy path instead.

The field interpolation and the deposit are written out in the loop rather than called:
Pyccel passes arrays to a function as Fortran descriptors, rebuilt at every call, which
costs more than the interpolation itself.
"""

# mypy: ignore-errors

from math import floor, hypot, sqrt


def node_cell(x: float, nodes: "float[:]") -> int:
    """The edge ``[nodes[c], nodes[c + 1]]`` of sorted ``nodes`` that holds ``x`` (the ends' beyond)."""
    n = nodes.shape[0]
    lo = 0
    hi = n - 1
    if x >= nodes[n - 1]:
        lo = n - 2
    elif x > nodes[0]:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if nodes[mid] <= x:
                lo = mid
            else:
                hi = mid
    return lo


def uniform_x(x: float, nodes: "float[:]", x0: float, h: float) -> float:
    """The uniform coordinate ``x0 + h * index`` of ``x`` on the graded axis of ``nodes``."""
    c = node_cell(x, nodes)
    return x0 + h * (c + (x - nodes[c]) / (nodes[c + 1] - nodes[c]))


def trace_trajectories_2d(
    active: "int[:]",
    x: "float[:,:]",
    v: "float[:,:]",
    qm: "float[:]",
    dq: "float[:]",
    steps: "int[:]",
    status: "int[:]",
    electrode: "int[:]",
    fxs: "float[:]",
    fys: "float[:]",
    patch_int: "int[:,:]",
    patch_real: "float[:,:]",
    origin_x: float,
    origin_y: float,
    h: float,
    upper_x: float,
    upper_y: float,
    cylindrical: int,
    integrator: int,
    step_fraction: float,
    max_steps: int,
    cells: "int[:,:]",
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
    graded: int,
    x_nodes: "float[:]",
) -> None:
    """Trace the ``active`` trajectories until they end, or reach a surface.

    ``x``, ``v`` ``(3, N)`` are the states (updated in place), ``qm`` the charge over
    mass, ``dq`` the signed current deposited per unit time, ``steps`` the steps
    taken so far. A step is that of `BeamTracer`, the Runge--Kutta step (``integrator``
    0) or Boris's kick--drift--kick (1, one field evaluation a step: the end's is the next
    start's), through the edge fields of patches (as ``staggered_field_2d``); it deposits
    ``dq dt`` at its Hermite midpoint (as ``deposit_trajectories_2d``) into ``charge[t]``,
    one slab per thread ``t < num_threads``, and records every crossing of the ``planes`` in
    ``hits[k, plane, c]`` (point, velocity, direction, step) for the first
    ``hits.shape[2]`` crossings, counting all in ``counts[k, plane]``. ``status`` ends
    as 0-4 (transmitted, backward, wall, electrode, max steps), or -2 for an end point
    in a sub-cell of ``subcells`` that an electrode surface cuts: the caller decides
    those with the electrodes' own test and resumes the others. ``cells`` gives each
    cell its block of ``refine**2`` sub-cells (-1: no electrode near), whose entry is
    the electrode containing the sub-cell, -1 for none or -2. A trajectory with
    ``path_slot[k] >= 0`` appends the mesh coordinates ``(x, y)`` or ``(x, r)`` of every
    step's end to ``paths[slot]``, as far as it holds, counting all in
    ``path_count[slot]``. Thread ``t`` traces every ``num_threads``-th active trajectory.
    ``cylindrical`` is the geometry: 0 planar, 1 cylindrical (fields of ``(x, r)``), 2 planar
    with a symmetry plane at ``y = origin_y`` (fields of ``(x, |y - origin_y|)``, ``E_y``
    mirrored, only ``upper_y`` an exit).

    With ``graded`` the ``x`` axis is graded: ``x_nodes`` holds the mesh's node positions
    along it, and every ``x`` is located by its uniform coordinate ``origin_x + h * index``
    (the patches' origins, spacings and extents along ``x`` are given in it, as are
    ``origin_x`` and ``upper_x``); a step takes the shorter of the patch's spacing and the
    local edge along ``x``. The trajectories, planes and paths stay in ``x``.

    The field and the charge live on patches, finest first and the whole mesh last:
    ``patch_int[p]`` holds a patch's node counts ``nx, ny``, the offsets of its ``fx``
    ``(nx + 1, ny)`` in ``fxs`` and ``fy`` ``(nx, ny + 1)`` in ``fys``, its node count and
    the offset of its nodes in ``charge[t]``; ``patch_real[p]`` its origin, node spacing and
    node extent ``x0, y0, x1, y1``. A point takes the finest patch containing it, and a
    step the spacing of the patch at its start. The mesh's ``origin``, ``h`` and
    ``upper`` (the last patch's) decide the exits and the electrode cells.
    """
    n_patches = patch_int.shape[0]
    nx = patch_int[n_patches - 1, 0]
    ny = patch_int[n_patches - 1, 1]
    n_active = active.shape[0]
    n_planes = planes.shape[0]
    capacity = hits.shape[2]
    path_capacity = paths.shape[1]
    # fmt: off
    #$ omp parallel for schedule(static, 1) num_threads(num_threads) private(lx, lq, cg, pp, qq, pnx, pny, ofx, ofy, ofn, pox, poy, phh, hloc, rr, t, m, k, slot, nstage, have, ca0, ca1, ca2, x0, x1, x2, v0, v1, v2, qmk, ns, st, stage, q0, q1, q2, qy, r, cy, cz, sx, sy, s, w, u, i, j, low, high, ex, ey, g0, g1, g2, a0, a1, a2, kx0, kx1, kx2, sx0, sx1, sx2, sv0, sv1, sv2, speed, accel, dt, d2, hdt, dt6, y0, y1, y2, w0, w1, w2, m0, m1, m2, deposit, p, xp, before, after, c, px, py, cell, si, sj, e)
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
            nstage = 4 if integrator == 0 else 2
            have = 0
            ca0 = 0.0
            ca1 = 0.0
            ca2 = 0.0
            while ns < max_steps:
                # the spacing of the finest patch at the step's start
                rr = x1
                if cylindrical == 1:
                    rr = hypot(x1, x2)
                elif cylindrical == 2:
                    rr = origin_y + abs(x1 - origin_y)
                lx = x0
                if graded != 0:
                    lx = uniform_x(x0, x_nodes, origin_x, h)
                hloc = h
                for qq in range(n_patches - 1):
                    if (
                        lx >= patch_real[qq, 3] and lx <= patch_real[qq, 5]
                        and rr >= patch_real[qq, 4] and rr <= patch_real[qq, 6]
                    ):  # fmt: skip
                        hloc = patch_real[qq, 2]
                        break
                if graded != 0:
                    # the patch's cells are shorter along x where the base edge is
                    cg = node_cell(x0, x_nodes)
                    d2 = (x_nodes[cg + 1] - x_nodes[cg]) / h
                    if d2 < 1.0:
                        hloc = hloc * d2
                # classical Runge--Kutta as BeamTracer._rk4: stage s samples the field at
                # q, its velocity is kx; sx and sv sum the stages with weights 1, 2, 2, 1
                q0 = x0
                q1 = x1
                q2 = x2
                kx0 = v0
                kx1 = v1
                kx2 = v2
                for stage in range(nstage):
                    # the acceleration at q (planar E(x, y), cylindrical E(x, r)), or for
                    # Boris at the start that of the last step's end
                    if stage == 0 and have != 0:
                        g0 = ca0
                        g1 = ca1
                        g2 = ca2
                    else:
                        qy = q1
                        cy = 0.0
                        cz = 0.0
                        if cylindrical == 1:
                            r = hypot(q1, q2)
                            if r > 0.0:
                                cy = q1 / r
                                cz = q2 / r
                            qy = r
                        elif cylindrical == 2:
                            # the mirrored field below the symmetry plane: E_y changes sign
                            cy = 1.0
                            if q1 < origin_y:
                                cy = -1.0
                            qy = origin_y + abs(q1 - origin_y)
                        lq = q0
                        if graded != 0:
                            lq = uniform_x(q0, x_nodes, origin_x, h)
                        pp = n_patches - 1
                        for qq in range(n_patches - 1):
                            if (
                                lq >= patch_real[qq, 3] and lq <= patch_real[qq, 5]
                                and qy >= patch_real[qq, 4] and qy <= patch_real[qq, 6]
                            ):  # fmt: skip
                                pp = qq
                                break
                        pnx = patch_int[pp, 0]
                        pny = patch_int[pp, 1]
                        ofx = patch_int[pp, 2]
                        ofy = patch_int[pp, 3]
                        pox = patch_real[pp, 0]
                        poy = patch_real[pp, 1]
                        phh = patch_real[pp, 2]
                        sx = (lq - pox) / phh
                        sy = (qy - poy) / phh
                        s = sx + 0.5
                        i = floor(s)
                        if i < 0:
                            i = 0
                        elif i > pnx - 1:
                            i = pnx - 1
                        j = floor(sy)
                        if j < 0:
                            j = 0
                        elif j > pny - 2:
                            j = pny - 2
                        u = s - i
                        w = sy - j
                        # fx (pnx + 1, pny) flattened
                        c = ofx + i * pny + j
                        low = fxs[c] + w * (fxs[c + 1] - fxs[c])
                        high = fxs[c + pny] + w * (fxs[c + pny + 1] - fxs[c + pny])
                        ex = low + u * (high - low)
                        s = sy + 0.5
                        i = floor(sx)
                        if i < 0:
                            i = 0
                        elif i > pnx - 2:
                            i = pnx - 2
                        j = floor(s)
                        if j < 0:
                            j = 0
                        elif j > pny - 1:
                            j = pny - 1
                        u = sx - i
                        w = s - j
                        # fy (pnx, pny + 1) flattened
                        c = ofy + i * (pny + 1) + j
                        low = fys[c] + w * (fys[c + 1] - fys[c])
                        high = fys[c + pny + 1] + w * (
                            fys[c + pny + 2] - fys[c + pny + 1]
                        )
                        ey = low + u * (high - low)
                        if cylindrical != 0:
                            g0 = qmk * ex
                            g1 = qmk * (ey * cy)
                            g2 = qmk * (ey * cz)
                        else:
                            g0 = qmk * ex
                            g1 = qmk * ey
                            g2 = qmk * 0.0
                    if stage == 0:
                        a0 = g0
                        a1 = g1
                        a2 = g2
                        speed = sqrt(v0 * v0 + v1 * v1 + v2 * v2)
                        accel = sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                        dt = -1.0
                        if speed > 0.0:
                            dt = hloc / speed
                        if accel > 0.0:
                            d2 = sqrt(2.0 * hloc / accel)
                            if dt < 0.0 or d2 < dt:
                                dt = d2
                        if dt < 0.0:
                            dt = hloc
                        dt = step_fraction * dt
                        hdt = 0.5 * dt
                        dt6 = dt / 6.0
                        if integrator != 0:
                            # Boris: the half kick, then the drift to the end of the step
                            kx0 = v0 + hdt * a0
                            kx1 = v1 + hdt * a1
                            kx2 = v2 + hdt * a2
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
                        # Boris: the second half kick at the end of the step
                        w0 = kx0 + hdt * g0
                        w1 = kx1 + hdt * g1
                        w2 = kx2 + hdt * g2
                        ca0 = g0
                        ca1 = g1
                        ca2 = g2
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

                # space charge: dq dt at the Hermite midpoint, area-weighted
                m0 = 0.5 * (x0 + y0) + dt * (v0 - w0) / 8.0
                m1 = 0.5 * (x1 + y1) + dt * (v1 - w1) / 8.0
                m2 = 0.5 * (x2 + y2) + dt * (v2 - w2) / 8.0
                if cylindrical == 1:
                    m1 = hypot(m1, m2)
                elif cylindrical == 2:
                    m1 = origin_y + abs(m1 - origin_y)
                if graded != 0:
                    m0 = uniform_x(m0, x_nodes, origin_x, h)
                pp = n_patches - 1
                for qq in range(n_patches - 1):
                    if (
                        m0 >= patch_real[qq, 3] and m0 <= patch_real[qq, 5]
                        and m1 >= patch_real[qq, 4] and m1 <= patch_real[qq, 6]
                    ):  # fmt: skip
                        pp = qq
                        break
                pnx = patch_int[pp, 0]
                pny = patch_int[pp, 1]
                ofn = patch_int[pp, 5]
                pox = patch_real[pp, 0]
                poy = patch_real[pp, 1]
                phh = patch_real[pp, 2]
                i = floor((m0 - pox) / phh)
                j = floor((m1 - poy) / phh)
                u = (m0 - (pox + phh * i)) / phh
                if cylindrical == 1:
                    low = poy + phh * j
                    high = low + phh
                    w = (m1 * m1 - low * low) / (high * high - low * low)
                else:
                    w = (m1 - (poy + phh * j)) / phh
                if i < 0:
                    u = 0.0
                    i = 0
                elif i >= pnx - 1:
                    u = 1.0
                    i = pnx - 2
                if j < 0:
                    w = 0.0
                    j = 0
                elif j >= pny - 1:
                    w = 1.0
                    j = pny - 2
                deposit = dq[k] * dt
                c = ofn + i * pny + j
                charge[t, c] += (1.0 - u) * (1.0 - w) * deposit
                charge[t, c + 1] += (1.0 - u) * w * deposit
                charge[t, c + pny] += u * (1.0 - w) * deposit
                charge[t, c + pny + 1] += u * w * deposit
                ns = ns + 1

                for p in range(n_planes):
                    xp = planes[p]
                    before = x0 - xp
                    after = y0 - xp
                    if (before < 0.0) != (after < 0.0):
                        s = before / (before - after)
                        c = counts[k, p]
                        if c < capacity:
                            hits[k, p, c, 0] = xp
                            hits[k, p, c, 1] = x1 + s * (y1 - x1)
                            hits[k, p, c, 2] = x2 + s * (y2 - x2)
                            hits[k, p, c, 3] = v0 + s * (w0 - v0)
                            hits[k, p, c, 4] = v1 + s * (w1 - v1)
                            hits[k, p, c, 5] = v2 + s * (w2 - v2)
                            hits[k, p, c, 6] = 1.0 if after >= 0.0 else -1.0
                            hits[k, p, c, 7] = 1.0 * ns
                        counts[k, p] = c + 1
                x0 = y0
                x1 = y1
                x2 = y2
                v0 = w0
                v1 = w1
                v2 = w2

                # the end of the step: off the mesh or in an electrode
                px = x0
                py = x1
                if cylindrical == 1:
                    py = hypot(x1, x2)
                elif cylindrical == 2:
                    py = origin_y + abs(x1 - origin_y)
                if slot >= 0:
                    c = path_count[slot]
                    if c < path_capacity:
                        paths[slot, c, 0] = px
                        paths[slot, c, 1] = py
                    path_count[slot] = c + 1
                if graded != 0:
                    px = uniform_x(px, x_nodes, origin_x, h)
                if px >= upper_x:
                    st = 0
                elif px < origin_x:
                    st = 1
                elif py > upper_y or (cylindrical == 0 and py < origin_y):
                    st = 2
                else:
                    i = int((px - origin_x) / h)
                    if i > nx - 2:
                        i = nx - 2
                    j = int((py - origin_y) / h)
                    if j > ny - 2:
                        j = ny - 2
                    cell = cells[i, j]
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
                        e = subcells[(cell * refine + si) * refine + sj]
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
