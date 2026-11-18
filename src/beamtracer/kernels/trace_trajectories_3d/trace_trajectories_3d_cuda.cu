// The CUDA version (GPU) of the trace_trajectories_3d kernel; trace_trajectories_3d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of active trajectories: one thread per trajectory, depositing atomically
// into charge(0, :) (num_threads is the number of host threads and is ignored here).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"

__device__ static long long clamp_index(long long i, long long high) {
    return i < 0 ? 0 : (i > high ? high : i);
}

// Trilinear interpolation of f at the fractional index (s, t, u), the cell clamped to
// [0, n - 2] on each axis.
__device__ static double trilinear(const Array3D<double>& f, double s, double t, double u) {
    const long long i = clamp_index((long long)floor(s), f.shape[0] - 2);
    const long long j = clamp_index((long long)floor(t), f.shape[1] - 2);
    const long long l = clamp_index((long long)floor(u), f.shape[2] - 2);
    const double a = s - i, b = t - j, c = u - l;
    return (1.0 - a) * ((1.0 - b) * ((1.0 - c) * f(i, j, l) + c * f(i, j, l + 1))
                        + b * ((1.0 - c) * f(i, j + 1, l) + c * f(i, j + 1, l + 1)))
         + a * ((1.0 - b) * ((1.0 - c) * f(i + 1, j, l) + c * f(i + 1, j, l + 1))
                + b * ((1.0 - c) * f(i + 1, j + 1, l) + c * f(i + 1, j + 1, l + 1)));
}

// B component n0 on the nodes (3, nx, ny, nz) at the fractional index (s, t, u).
__device__ static double node_field(const Array4D<double>& bn, long long n0, double s,
                                    double t, double u) {
    const long long i = clamp_index((long long)floor(s), bn.shape[1] - 2);
    const long long j = clamp_index((long long)floor(t), bn.shape[2] - 2);
    const long long l = clamp_index((long long)floor(u), bn.shape[3] - 2);
    const double a = s - i, b = t - j, c = u - l;
    return (1.0 - a) * ((1.0 - b) * ((1.0 - c) * bn(n0, i, j, l) + c * bn(n0, i, j, l + 1))
                        + b * ((1.0 - c) * bn(n0, i, j + 1, l) + c * bn(n0, i, j + 1, l + 1)))
         + a * ((1.0 - b) * ((1.0 - c) * bn(n0, i + 1, j, l) + c * bn(n0, i + 1, j, l + 1))
                + b * ((1.0 - c) * bn(n0, i + 1, j + 1, l) + c * bn(n0, i + 1, j + 1, l + 1)));
}

// E and B at (q0, q1, q2): below a symmetry plane those of the mirror image, E_y (E_z) and
// the other two components of B changing sign.
__device__ static void fields_3d(
    double q0, double q1, double q2, const Array3D<double>& fx, const Array3D<double>& fy,
    const Array3D<double>& fz, const Array4D<double>& bn, long long has_b, double origin_x,
    double origin_y, double origin_z, double h, long long mirror_y, long long mirror_z,
    double* e, double* b
) {
    double s1 = 1.0, s2 = 1.0;
    if (mirror_y != 0 && q1 < origin_y) {
        q1 = 2.0 * origin_y - q1;
        s1 = -1.0;
    }
    if (mirror_z != 0 && q2 < origin_z) {
        q2 = 2.0 * origin_z - q2;
        s2 = -1.0;
    }
    const double sx = (q0 - origin_x) / h, sy = (q1 - origin_y) / h, sz = (q2 - origin_z) / h;
    // the ghost edges make the staggered grids one longer: clamp to their own ends
    e[0] = trilinear(fx, sx + 0.5, sy, sz);
    e[1] = s1 * trilinear(fy, sx, sy + 0.5, sz);
    e[2] = s2 * trilinear(fz, sx, sy, sz + 0.5);
    b[0] = b[1] = b[2] = 0.0;
    if (has_b != 0) {
        b[0] = s1 * s2 * node_field(bn, 0, sx, sy, sz);
        b[1] = s2 * node_field(bn, 1, sx, sy, sz);
        b[2] = s1 * node_field(bn, 2, sx, sy, sz);
    }
}

// Boris's kick--rotate--kick over a half step: tau = qm dt / 4.
__device__ static void boris_half(const double* u_in, const double* e, const double* b,
                                  double tau, double* u_out) {
    double u0 = u_in[0] + tau * e[0], u1 = u_in[1] + tau * e[1], u2 = u_in[2] + tau * e[2];
    const double t0 = tau * b[0], t1 = tau * b[1], t2 = tau * b[2];
    const double tt = 2.0 / (1.0 + t0 * t0 + t1 * t1 + t2 * t2);
    const double r0 = u0 + u1 * t2 - u2 * t1;
    const double r1 = u1 + u2 * t0 - u0 * t2;
    const double r2 = u2 + u0 * t1 - u1 * t0;
    u0 += tt * (r1 * t2 - r2 * t1);
    u1 += tt * (r2 * t0 - r0 * t2);
    u2 += tt * (r0 * t1 - r1 * t0);
    u_out[0] = u0 + tau * e[0];
    u_out[1] = u1 + tau * e[1];
    u_out[2] = u2 + tau * e[2];
}

__device__ static void lorentz(double qm, const double* e, const double* b, const double* u,
                               double* g) {
    g[0] = qm * (e[0] + u[1] * b[2] - u[2] * b[1]);
    g[1] = qm * (e[1] + u[2] * b[0] - u[0] * b[2]);
    g[2] = qm * (e[2] + u[0] * b[1] - u[1] * b[0]);
}

extern "C" __global__
void trace_trajectories_3d(
    Array1D<long long> active,
    Array2D<double> x,
    Array2D<double> v,
    Array1D<double> qm,
    Array1D<double> dq,
    Array1D<long long> steps,
    Array1D<long long> status,
    Array1D<long long> electrode,
    Array3D<double> fx,
    Array3D<double> fy,
    Array3D<double> fz,
    Array4D<double> bn,
    long long has_b,
    double origin_x,
    double origin_y,
    double origin_z,
    double h,
    double upper_x,
    double upper_y,
    double upper_z,
    long long mirror_y,
    long long mirror_z,
    long long integrator,
    double step_fraction,
    long long max_steps,
    Array3D<long long> cells,
    Array1D<long long> subcells,
    long long refine,
    Array1D<double> planes,
    Array4D<double> hits,
    Array2D<long long> counts,
    Array1D<long long> path_slot,
    Array3D<double> paths,
    Array1D<long long> path_count,
    long long num_threads,
    Array2D<double> charge
) {
    const long long m = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (m >= active.shape[0]) return;
    const long long nx = fy.shape[0], ny = fx.shape[1], nz = fx.shape[2];
    const long long n_planes = planes.shape[0];
    const long long capacity = hits.shape[2];
    const long long k = active(m);
    const long long slot = path_slot(k);
    const long long path_capacity = paths.shape[1];
    double xs[3] = {x(0, k), x(1, k), x(2, k)};
    double vs[3] = {v(0, k), v(1, k), v(2, k)};
    const double qmk = qm(k);
    const double dqk = dq(k);
    long long ns = steps(k);
    long long st = -1;
    bool have = false;  // Boris: the fields at the start are the last step's end's
    double e[3], b[3];
    while (ns < max_steps) {
        if (!have) {
            fields_3d(xs[0], xs[1], xs[2], fx, fy, fz, bn, has_b, origin_x, origin_y,
                      origin_z, h, mirror_y, mirror_z, e, b);
        }
        double a[3];
        lorentz(qmk, e, b, vs, a);
        const double speed = sqrt(vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2]);
        const double accel = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double dt = -1.0;
        if (speed > 0.0) dt = h / speed;
        if (accel > 0.0) {
            const double d2 = sqrt(2.0 * h / accel);
            if (dt < 0.0 || d2 < dt) dt = d2;
        }
        if (dt < 0.0) dt = h;
        dt *= step_fraction;
        const double hdt = 0.5 * dt;
        double y[3], w[3];
        if (integrator != 0) {
            const double tau = 0.5 * hdt * qmk;
            double half[3];
            boris_half(vs, e, b, tau, half);
            for (int d = 0; d < 3; ++d) y[d] = xs[d] + dt * half[d];
            fields_3d(y[0], y[1], y[2], fx, fy, fz, bn, has_b, origin_x, origin_y, origin_z,
                      h, mirror_y, mirror_z, e, b);
            boris_half(half, e, b, tau, w);
            have = true;
        } else {
            double q[3], ku[3], g[3], sx[3], sv[3], ee[3], bb[3];
            for (int d = 0; d < 3; ++d) {
                sx[d] = vs[d];
                sv[d] = a[d];
                q[d] = xs[d] + hdt * vs[d];
                ku[d] = vs[d] + hdt * a[d];
            }
            for (int stage = 1; stage < 4; ++stage) {
                fields_3d(q[0], q[1], q[2], fx, fy, fz, bn, has_b, origin_x, origin_y,
                          origin_z, h, mirror_y, mirror_z, ee, bb);
                lorentz(qmk, ee, bb, ku, g);
                const double weight = stage < 3 ? 2.0 : 1.0;
                const double step = stage == 1 ? hdt : dt;
                for (int d = 0; d < 3; ++d) {
                    sx[d] += weight * ku[d];
                    sv[d] += weight * g[d];
                }
                if (stage < 3) {
                    for (int d = 0; d < 3; ++d) {
                        q[d] = xs[d] + step * ku[d];
                        ku[d] = vs[d] + step * g[d];
                    }
                }
            }
            for (int d = 0; d < 3; ++d) {
                y[d] = xs[d] + dt / 6.0 * sx[d];
                w[d] = vs[d] + dt / 6.0 * sv[d];
            }
        }

        // space charge: dq dt at the Hermite midpoint, trilinear
        double mid[3];
        for (int d = 0; d < 3; ++d) mid[d] = 0.5 * (xs[d] + y[d]) + dt * (vs[d] - w[d]) / 8.0;
        if (mirror_y != 0 && mid[1] < origin_y) mid[1] = 2.0 * origin_y - mid[1];
        if (mirror_z != 0 && mid[2] < origin_z) mid[2] = 2.0 * origin_z - mid[2];
        const double origin[3] = {origin_x, origin_y, origin_z};
        const long long n[3] = {nx, ny, nz};
        long long idx[3];
        double frac[3];
        for (int d = 0; d < 3; ++d) {
            const double s = (mid[d] - origin[d]) / h;
            idx[d] = (long long)floor(s);
            frac[d] = s - idx[d];
            if (idx[d] < 0) {
                frac[d] = 0.0;
                idx[d] = 0;
            } else if (idx[d] >= n[d] - 1) {
                frac[d] = 1.0;
                idx[d] = n[d] - 2;
            }
        }
        const double deposit = dqk * dt;
        const long long c = (idx[0] * ny + idx[1]) * nz + idx[2];
        for (int di = 0; di < 2; ++di) {
            const double wi = di ? frac[0] : 1.0 - frac[0];
            for (int dj = 0; dj < 2; ++dj) {
                const double wj = dj ? frac[1] : 1.0 - frac[1];
                for (int dk = 0; dk < 2; ++dk) {
                    const double wk = dk ? frac[2] : 1.0 - frac[2];
                    cunumpy_atomic_add(&charge(0, c + di * ny * nz + dj * nz + dk),
                                       wi * wj * wk * deposit);
                }
            }
        }
        ns += 1;

        for (long long p = 0; p < n_planes; ++p) {
            const double xp = planes(p);
            const double before = xs[0] - xp, after = y[0] - xp;
            if ((before < 0.0) != (after < 0.0)) {
                const double s = before / (before - after);
                const long long cc = counts(k, p);
                if (cc < capacity) {
                    hits(k, p, cc, 0) = xp;
                    hits(k, p, cc, 1) = xs[1] + s * (y[1] - xs[1]);
                    hits(k, p, cc, 2) = xs[2] + s * (y[2] - xs[2]);
                    hits(k, p, cc, 3) = vs[0] + s * (w[0] - vs[0]);
                    hits(k, p, cc, 4) = vs[1] + s * (w[1] - vs[1]);
                    hits(k, p, cc, 5) = vs[2] + s * (w[2] - vs[2]);
                    hits(k, p, cc, 6) = after >= 0.0 ? 1.0 : -1.0;
                    hits(k, p, cc, 7) = 1.0 * ns;
                }
                counts(k, p) = cc + 1;
            }
        }
        for (int d = 0; d < 3; ++d) {
            xs[d] = y[d];
            vs[d] = w[d];
        }

        // the end of the step: off the mesh or in an electrode
        const double px = xs[0];
        double py = xs[1], pz = xs[2];
        if (mirror_y != 0 && py < origin_y) py = 2.0 * origin_y - py;
        if (mirror_z != 0 && pz < origin_z) pz = 2.0 * origin_z - pz;
        if (slot >= 0) {
            const long long cc = path_count(slot);
            if (cc < path_capacity) {
                paths(slot, cc, 0) = px;
                paths(slot, cc, 1) = py;
                paths(slot, cc, 2) = pz;
            }
            path_count(slot) = cc + 1;
        }
        if (px >= upper_x) {
            st = 0;
        } else if (px < origin_x) {
            st = 1;
        } else if (py > upper_y || py < origin_y || pz > upper_z || pz < origin_z) {
            st = 2;
        } else {
            const long long i = min((long long)((px - origin_x) / h), nx - 2);
            const long long j = min((long long)((py - origin_y) / h), ny - 2);
            const long long l = min((long long)((pz - origin_z) / h), nz - 2);
            const long long cell = cells(i, j, l);
            if (cell >= 0) {
                const long long si = clamp_index((long long)(((px - origin_x) / h - i) * refine), refine - 1);
                const long long sj = clamp_index((long long)(((py - origin_y) / h - j) * refine), refine - 1);
                const long long sk = clamp_index((long long)(((pz - origin_z) / h - l) * refine), refine - 1);
                const long long en = subcells(((cell * refine + si) * refine + sj) * refine + sk);
                if (en >= 0) {
                    st = 3;
                    electrode(k) = en;
                } else if (en == -2) {
                    st = -2;
                }
            }
        }
        if (st != -1) break;
    }
    if (st == -1) st = 4;
    x(0, k) = xs[0];
    x(1, k) = xs[1];
    x(2, k) = xs[2];
    v(0, k) = vs[0];
    v(1, k) = vs[1];
    v(2, k) = vs[2];
    steps(k) = ns;
    status(k) = st;
}
