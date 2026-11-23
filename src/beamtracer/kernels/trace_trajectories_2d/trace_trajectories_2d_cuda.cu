// The CUDA version (GPU) of the trace_trajectories_2d kernel; trace_trajectories_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of active trajectories: one thread per trajectory, depositing atomically
// into charge(0, :, :) (num_threads is the number of host threads and is ignored here).
// With inv_c2 = 1 / c^2 > 0 the motion is relativistic: v holds the proper velocity
// u = gamma v and the positions move with u / gamma (mpiparticles_lorentz_factor).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "mpiparticles/kernels/boris.cuh"

// The finest patch containing (px, py): patches finest first, the whole mesh last.
__device__ static long long patch_of(
    double px, double py, const Array2D<double>& patch_real, long long n_patches
) {
    for (long long q = 0; q < n_patches - 1; ++q) {
        if (px >= patch_real(q, 3) && px <= patch_real(q, 5) &&
            py >= patch_real(q, 4) && py <= patch_real(q, 6)) {
            return q;
        }
    }
    return n_patches - 1;
}

__device__ static void edge_field_2d(
    double px, double py, const Array1D<double>& fxs, const Array1D<double>& fys,
    const Array2D<long long>& patch_int, const Array2D<double>& patch_real,
    long long n_patches, double* ex, double* ey
) {
    const long long pp = patch_of(px, py, patch_real, n_patches);
    const long long nx = patch_int(pp, 0);
    const long long ny = patch_int(pp, 1);
    const double sx = (px - patch_real(pp, 0)) / patch_real(pp, 2);
    const double sy = (py - patch_real(pp, 1)) / patch_real(pp, 2);
    const double s = sx + 0.5;
    long long i = min(max((long long)floor(s), 0LL), nx - 1);
    long long j = min(max((long long)floor(sy), 0LL), ny - 2);
    double u = s - i;
    double w = sy - j;
    long long c = patch_int(pp, 2) + i * ny + j;  // fx (nx + 1, ny)
    double low = fxs(c) + w * (fxs(c + 1) - fxs(c));
    double high = fxs(c + ny) + w * (fxs(c + ny + 1) - fxs(c + ny));
    *ex = low + u * (high - low);
    const double t = sy + 0.5;
    i = min(max((long long)floor(sx), 0LL), nx - 2);
    j = min(max((long long)floor(t), 0LL), ny - 1);
    u = sx - i;
    w = t - j;
    c = patch_int(pp, 3) + i * (ny + 1) + j;  // fy (nx, ny + 1)
    low = fys(c) + w * (fys(c + 1) - fys(c));
    high = fys(c + ny + 1) + w * (fys(c + ny + 2) - fys(c + ny + 1));
    *ey = low + u * (high - low);
}

// The mesh coordinate of (x1, x2) across the axis: y (planar), r (cylindrical) or y
// mirrored onto the mesh side of the symmetry plane at plane_y (planar, geometry 2).
__device__ static double mesh_y(double x1, double x2, long long cylindrical, double plane_y) {
    if (cylindrical == 1) {
        return hypot(x1, x2);
    }
    if (cylindrical == 2) {
        return plane_y + fabs(x1 - plane_y);
    }
    return x1;
}

// the edge [nodes(c), nodes(c + 1)] of the sorted nodes that holds x (the ends' beyond)
__device__ static long long node_cell(double x, const Array1D<double>& nodes) {
    const long long n = nodes.shape[0];
    long long lo = 0, hi = n - 1;
    if (x >= nodes(n - 1)) {
        lo = n - 2;
    } else if (x > nodes(0)) {
        while (hi - lo > 1) {
            const long long mid = (lo + hi) / 2;
            if (nodes(mid) <= x) lo = mid; else hi = mid;
        }
    }
    return lo;
}

// the uniform coordinate x0 + h * index of x on the graded axis of the nodes
__device__ static double uniform_x(double x, long long graded, const Array1D<double>& nodes,
                                   double x0, double h) {
    if (graded == 0) return x;
    const long long c = node_cell(x, nodes);
    return x0 + h * (c + (x - nodes(c)) / (nodes(c + 1) - nodes(c)));
}

__device__ static void acceleration_2d(
    double x0, double x1, double x2, double qm, const Array1D<double>& fxs,
    const Array1D<double>& fys, const Array2D<long long>& patch_int,
    const Array2D<double>& patch_real, long long n_patches,
    long long cylindrical, double plane_y, long long graded,
    const Array1D<double>& x_nodes, double origin_x, double h,
    double* a0, double* a1, double* a2
) {
    double ex, ey;
    x0 = uniform_x(x0, graded, x_nodes, origin_x, h);
    if (cylindrical != 1) {
        // planar; below a symmetry plane the field is mirrored: E_y changes sign
        const double sign = (cylindrical == 2 && x1 < plane_y) ? -1.0 : 1.0;
        edge_field_2d(x0, mesh_y(x1, x2, cylindrical, plane_y), fxs, fys, patch_int,
                      patch_real, n_patches, &ex, &ey);
        *a0 = qm * ex;
        *a1 = qm * (sign * ey);
        *a2 = qm * 0.0;
        return;
    }
    const double r = hypot(x1, x2);
    double cy = 0.0, cz = 0.0;
    if (r > 0.0) {
        cy = x1 / r;
        cz = x2 / r;
    }
    edge_field_2d(x0, r, fxs, fys, patch_int, patch_real, n_patches, &ex, &ey);
    *a0 = qm * ex;
    *a1 = qm * (ey * cy);
    *a2 = qm * (ey * cz);
}

__device__ static void deposit_point_2d(
    double x, double y, double value, const Array2D<long long>& patch_int,
    const Array2D<double>& patch_real, long long n_patches, long long cylindrical,
    Array2D<double>& charge
) {
    const long long pp = patch_of(x, y, patch_real, n_patches);
    const long long nx = patch_int(pp, 0);
    const long long ny = patch_int(pp, 1);
    const double origin_x = patch_real(pp, 0);
    const double origin_y = patch_real(pp, 1);
    const double h = patch_real(pp, 2);
    long long i = (long long)floor((x - origin_x) / h);
    long long j = (long long)floor((y - origin_y) / h);
    double a = (x - (origin_x + h * i)) / h;
    double b;
    if (cylindrical == 1) {
        const double r1 = origin_y + h * j;
        const double r2 = r1 + h;
        b = (y * y - r1 * r1) / (r2 * r2 - r1 * r1);
    } else {
        b = (y - (origin_y + h * j)) / h;
    }
    if (i < 0) {
        a = 0.0;
        i = 0;
    } else if (i >= nx - 1) {
        a = 1.0;
        i = nx - 2;
    }
    if (j < 0) {
        b = 0.0;
        j = 0;
    } else if (j >= ny - 1) {
        b = 1.0;
        j = ny - 2;
    }
    const long long c = patch_int(pp, 5) + i * ny + j;
    cunumpy_atomic_add(&charge(0, c), (1.0 - a) * (1.0 - b) * value);
    cunumpy_atomic_add(&charge(0, c + 1), (1.0 - a) * b * value);
    cunumpy_atomic_add(&charge(0, c + ny), a * (1.0 - b) * value);
    cunumpy_atomic_add(&charge(0, c + ny + 1), a * b * value);
}

extern "C" __global__
void trace_trajectories_2d(
    Array1D<long long> active,
    Array2D<double> x,
    Array2D<double> v,
    Array1D<double> qm,
    Array1D<double> dq,
    Array1D<long long> steps,
    Array1D<long long> status,
    Array1D<long long> electrode,
    Array1D<double> fxs,
    Array1D<double> fys,
    Array2D<long long> patch_int,
    Array2D<double> patch_real,
    double origin_x,
    double origin_y,
    double h,
    double upper_x,
    double upper_y,
    long long cylindrical,
    long long integrator,
    double step_fraction,
    long long max_steps,
    Array2D<long long> cells,
    Array1D<long long> subcells,
    long long refine,
    Array1D<double> planes,
    Array4D<double> hits,
    Array2D<long long> counts,
    Array1D<long long> path_slot,
    Array3D<double> paths,
    Array1D<long long> path_count,
    long long num_threads,
    Array2D<double> charge,
    long long graded,
    Array1D<double> x_nodes,
    double inv_c2
) {
    const long long m = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (m >= active.shape[0]) return;
    const long long n_patches = patch_int.shape[0];
    const long long nx = patch_int(n_patches - 1, 0);
    const long long ny = patch_int(n_patches - 1, 1);
    const long long n_planes = planes.shape[0];
    const long long capacity = hits.shape[2];
    const long long k = active(m);
    const long long slot = path_slot(k);
    const long long path_capacity = paths.shape[1];
    double x0 = x(0, k), x1 = x(1, k), x2 = x(2, k);
    double v0 = v(0, k), v1 = v(1, k), v2 = v(2, k);
    const double qmk = qm(k);
    const double dqk = dq(k);
    long long ns = steps(k);
    long long st = -1;
    bool have = false;  // Boris: the acceleration at the start is the last step's end's
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    while (ns < max_steps) {
        // the spacing of the finest patch at the step's start
        double hloc = patch_real(
            patch_of(uniform_x(x0, graded, x_nodes, origin_x, h),
                     mesh_y(x1, x2, cylindrical, origin_y), patch_real, n_patches), 2);
        if (graded != 0) {
            // the patch's cells are shorter along x where the base edge is
            const long long cg = node_cell(x0, x_nodes);
            const double ratio = (x_nodes(cg + 1) - x_nodes(cg)) / h;
            if (ratio < 1.0) hloc = hloc * ratio;
        }
        if (!have) {
            acceleration_2d(x0, x1, x2, qmk, fxs, fys, patch_int, patch_real, n_patches, cylindrical, origin_y, graded, x_nodes, origin_x, h, &a0, &a1, &a2);
        }
        const double gv = 1.0 / mpiparticles_lorentz_factor(v0, v1, v2, inv_c2);
        const double speed = gv * sqrt(v0 * v0 + v1 * v1 + v2 * v2);
        const double accel = sqrt(a0 * a0 + a1 * a1 + a2 * a2);
        double dt = -1.0;
        if (speed > 0.0) dt = hloc / speed;
        if (accel > 0.0) {
            const double d2 = sqrt(2.0 * hloc / accel);
            if (dt < 0.0 || d2 < dt) dt = d2;
        }
        if (dt < 0.0) dt = hloc;
        dt = step_fraction * dt;
        double y0, y1, y2, w0, w1, w2;
        if (integrator != 0) {
            // Boris's kick--drift--kick, as BeamTracer._boris without a magnetic field
            const double hdt = 0.5 * dt;
            const double kx0 = v0 + hdt * a0;
            const double kx1 = v1 + hdt * a1;
            const double kx2 = v2 + hdt * a2;
            const double gk = 1.0 / mpiparticles_lorentz_factor(kx0, kx1, kx2, inv_c2);
            y0 = x0 + dt * gk * kx0;
            y1 = x1 + dt * gk * kx1;
            y2 = x2 + dt * gk * kx2;
            double g0, g1, g2;
            acceleration_2d(y0, y1, y2, qmk, fxs, fys, patch_int, patch_real, n_patches, cylindrical, origin_y, graded, x_nodes, origin_x, h, &g0, &g1, &g2);
            w0 = kx0 + hdt * g0;
            w1 = kx1 + hdt * g1;
            w2 = kx2 + hdt * g2;
            a0 = g0; a1 = g1; a2 = g2;
            have = true;
        } else {
        const double k2x0 = v0 + 0.5 * dt * a0;
        const double k2x1 = v1 + 0.5 * dt * a1;
        const double k2x2 = v2 + 0.5 * dt * a2;
        double k2v0, k2v1, k2v2;
        const double g2 = 1.0 / mpiparticles_lorentz_factor(k2x0, k2x1, k2x2, inv_c2);
        acceleration_2d(x0 + 0.5 * dt * gv * v0, x1 + 0.5 * dt * gv * v1, x2 + 0.5 * dt * gv * v2, qmk, fxs, fys, patch_int, patch_real, n_patches,
                        cylindrical, origin_y, graded, x_nodes, origin_x, h, &k2v0, &k2v1, &k2v2);
        const double k3x0 = v0 + 0.5 * dt * k2v0;
        const double k3x1 = v1 + 0.5 * dt * k2v1;
        const double k3x2 = v2 + 0.5 * dt * k2v2;
        double k3v0, k3v1, k3v2;
        const double g3 = 1.0 / mpiparticles_lorentz_factor(k3x0, k3x1, k3x2, inv_c2);
        acceleration_2d(x0 + 0.5 * dt * g2 * k2x0, x1 + 0.5 * dt * g2 * k2x1, x2 + 0.5 * dt * g2 * k2x2, qmk, fxs, fys, patch_int, patch_real, n_patches,
                        cylindrical, origin_y, graded, x_nodes, origin_x, h, &k3v0, &k3v1, &k3v2);
        const double k4x0 = v0 + dt * k3v0;
        const double k4x1 = v1 + dt * k3v1;
        const double k4x2 = v2 + dt * k3v2;
        double k4v0, k4v1, k4v2;
        const double g4 = 1.0 / mpiparticles_lorentz_factor(k4x0, k4x1, k4x2, inv_c2);
        acceleration_2d(x0 + dt * g3 * k3x0, x1 + dt * g3 * k3x1, x2 + dt * g3 * k3x2, qmk, fxs, fys, patch_int, patch_real, n_patches,
                        cylindrical, origin_y, graded, x_nodes, origin_x, h, &k4v0, &k4v1, &k4v2);
        y0 = x0 + dt / 6.0 * (gv * v0 + 2.0 * g2 * k2x0 + 2.0 * g3 * k3x0 + g4 * k4x0);
        y1 = x1 + dt / 6.0 * (gv * v1 + 2.0 * g2 * k2x1 + 2.0 * g3 * k3x1 + g4 * k4x1);
        y2 = x2 + dt / 6.0 * (gv * v2 + 2.0 * g2 * k2x2 + 2.0 * g3 * k3x2 + g4 * k4x2);
        w0 = v0 + dt / 6.0 * (a0 + 2.0 * k2v0 + 2.0 * k3v0 + k4v0);
        w1 = v1 + dt / 6.0 * (a1 + 2.0 * k2v1 + 2.0 * k3v1 + k4v1);
        w2 = v2 + dt / 6.0 * (a2 + 2.0 * k2v2 + 2.0 * k3v2 + k4v2);
        }
        const double gw = 1.0 / mpiparticles_lorentz_factor(w0, w1, w2, inv_c2);
        const double m0 = 0.5 * (x0 + y0) + dt * (gv * v0 - gw * w0) / 8.0;
        const double m1 = 0.5 * (x1 + y1) + dt * (gv * v1 - gw * w1) / 8.0;
        const double m2 = 0.5 * (x2 + y2) + dt * (gv * v2 - gw * w2) / 8.0;
        const double my = mesh_y(m1, m2, cylindrical, origin_y);
        deposit_point_2d(uniform_x(m0, graded, x_nodes, origin_x, h), my, dqk * dt, patch_int, patch_real, n_patches, cylindrical, charge);
        ns = ns + 1;
        for (long long p = 0; p < n_planes; ++p) {
            const double xp = planes(p);
            const double before = x0 - xp;
            const double after = y0 - xp;
            if ((before < 0.0) != (after < 0.0)) {
                const double s = before / (before - after);
                const long long c = counts(k, p);
                if (c < capacity) {
                    hits(k, p, c, 0) = xp;
                    hits(k, p, c, 1) = x1 + s * (y1 - x1);
                    hits(k, p, c, 2) = x2 + s * (y2 - x2);
                    hits(k, p, c, 3) = gv * v0 + s * (gw * w0 - gv * v0);
                    hits(k, p, c, 4) = gv * v1 + s * (gw * w1 - gv * v1);
                    hits(k, p, c, 5) = gv * v2 + s * (gw * w2 - gv * v2);
                    hits(k, p, c, 6) = after >= 0.0 ? 1.0 : -1.0;
                    hits(k, p, c, 7) = 1.0 * ns;
                }
                counts(k, p) = c + 1;
            }
        }
        x0 = y0; x1 = y1; x2 = y2;
        v0 = w0; v1 = w1; v2 = w2;
        const double px = x0;
        const double py = mesh_y(x1, x2, cylindrical, origin_y);
        if (slot >= 0) {
            const long long c = path_count(slot);
            if (c < path_capacity) {
                paths(slot, c, 0) = px;
                paths(slot, c, 1) = py;
            }
            path_count(slot) = c + 1;
        }
        const double lpx = uniform_x(px, graded, x_nodes, origin_x, h);
        if (lpx >= upper_x) {
            st = 0;
        } else if (lpx < origin_x) {
            st = 1;
        } else if (py > upper_y || (cylindrical == 0 && py < origin_y)) {
            st = 2;
        } else {
            const long long i = min(max((long long)((lpx - origin_x) / h), 0LL), nx - 2);
            const long long j = min(max((long long)((py - origin_y) / h), 0LL), ny - 2);
            const long long cell = cells(i, j);
            if (cell >= 0) {
                const long long si = min(max((long long)(((lpx - origin_x) / h - i) * refine), 0LL), refine - 1);
                const long long sj = min(max((long long)(((py - origin_y) / h - j) * refine), 0LL), refine - 1);
                const long long e = subcells((cell * refine + si) * refine + sj);
                if (e >= 0) {
                    st = 3;
                    electrode(k) = e;
                } else if (e == -2) {
                    st = -2;
                }
            }
        }
        if (st != -1) break;
    }
    if (st == -1) st = 4;
    x(0, k) = x0; x(1, k) = x1; x(2, k) = x2;
    v(0, k) = v0; v(1, k) = v1; v(2, k) = v2;
    steps(k) = ns;
    status(k) = st;
}
