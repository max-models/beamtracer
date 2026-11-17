// The CUDA version (GPU) of the trace_trajectories_2d kernel; trace_trajectories_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of active trajectories: one thread per trajectory, depositing atomically
// into charge(0, :, :) (num_threads is the number of host threads and is ignored here).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"

__device__ static void edge_field_2d(
    double px, double py, const Array2D<double>& fx, const Array2D<double>& fy,
    double origin_x, double origin_y, double h, double* ex, double* ey
) {
    const long long nx = fy.shape[0];
    const long long ny = fx.shape[1];
    const double sx = (px - origin_x) / h;
    const double sy = (py - origin_y) / h;
    const double s = sx + 0.5;
    long long i = min(max((long long)floor(s), 0LL), nx - 1);
    long long j = min(max((long long)floor(sy), 0LL), ny - 2);
    double u = s - i;
    double w = sy - j;
    double low = fx(i, j) + w * (fx(i, j + 1) - fx(i, j));
    double high = fx(i + 1, j) + w * (fx(i + 1, j + 1) - fx(i + 1, j));
    *ex = low + u * (high - low);
    const double t = sy + 0.5;
    i = min(max((long long)floor(sx), 0LL), nx - 2);
    j = min(max((long long)floor(t), 0LL), ny - 1);
    u = sx - i;
    w = t - j;
    low = fy(i, j) + w * (fy(i, j + 1) - fy(i, j));
    high = fy(i + 1, j) + w * (fy(i + 1, j + 1) - fy(i + 1, j));
    *ey = low + u * (high - low);
}

__device__ static void acceleration_2d(
    double x0, double x1, double x2, double qm, const Array2D<double>& fx,
    const Array2D<double>& fy, double origin_x, double origin_y, double h,
    long long cylindrical, double* a0, double* a1, double* a2
) {
    double ex, ey;
    if (cylindrical == 0) {
        edge_field_2d(x0, x1, fx, fy, origin_x, origin_y, h, &ex, &ey);
        *a0 = qm * ex;
        *a1 = qm * ey;
        *a2 = qm * 0.0;
        return;
    }
    const double r = hypot(x1, x2);
    double cy = 0.0, cz = 0.0;
    if (r > 0.0) {
        cy = x1 / r;
        cz = x2 / r;
    }
    edge_field_2d(x0, r, fx, fy, origin_x, origin_y, h, &ex, &ey);
    *a0 = qm * ex;
    *a1 = qm * (ey * cy);
    *a2 = qm * (ey * cz);
}

__device__ static void deposit_point_2d(
    double x, double y, double value, double origin_x, double origin_y, double h,
    long long cylindrical, Array3D<double>& charge
) {
    const long long nx = charge.shape[1];
    const long long ny = charge.shape[2];
    long long i = (long long)floor((x - origin_x) / h);
    long long j = (long long)floor((y - origin_y) / h);
    double a = (x - (origin_x + h * i)) / h;
    double b;
    if (cylindrical != 0) {
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
    cunumpy_atomic_add(&charge(0, i, j), (1.0 - a) * (1.0 - b) * value);
    cunumpy_atomic_add(&charge(0, i, j + 1), (1.0 - a) * b * value);
    cunumpy_atomic_add(&charge(0, i + 1, j), a * (1.0 - b) * value);
    cunumpy_atomic_add(&charge(0, i + 1, j + 1), a * b * value);
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
    Array2D<double> fx,
    Array2D<double> fy,
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
    Array3D<double> charge
) {
    const long long m = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (m >= active.shape[0]) return;
    const long long nx = fy.shape[0];
    const long long ny = fx.shape[1];
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
        if (!have) {
            acceleration_2d(x0, x1, x2, qmk, fx, fy, origin_x, origin_y, h, cylindrical, &a0, &a1, &a2);
        }
        const double speed = sqrt(v0 * v0 + v1 * v1 + v2 * v2);
        const double accel = sqrt(a0 * a0 + a1 * a1 + a2 * a2);
        double dt = -1.0;
        if (speed > 0.0) dt = h / speed;
        if (accel > 0.0) {
            const double d2 = sqrt(2.0 * h / accel);
            if (dt < 0.0 || d2 < dt) dt = d2;
        }
        if (dt < 0.0) dt = h;
        dt = step_fraction * dt;
        double y0, y1, y2, w0, w1, w2;
        if (integrator != 0) {
            // Boris's kick--drift--kick, as BeamTracer._boris without a magnetic field
            const double hdt = 0.5 * dt;
            const double kx0 = v0 + hdt * a0;
            const double kx1 = v1 + hdt * a1;
            const double kx2 = v2 + hdt * a2;
            y0 = x0 + dt * kx0;
            y1 = x1 + dt * kx1;
            y2 = x2 + dt * kx2;
            double g0, g1, g2;
            acceleration_2d(y0, y1, y2, qmk, fx, fy, origin_x, origin_y, h, cylindrical, &g0, &g1, &g2);
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
        acceleration_2d(x0 + 0.5 * dt * v0, x1 + 0.5 * dt * v1, x2 + 0.5 * dt * v2, qmk, fx, fy,
                        origin_x, origin_y, h, cylindrical, &k2v0, &k2v1, &k2v2);
        const double k3x0 = v0 + 0.5 * dt * k2v0;
        const double k3x1 = v1 + 0.5 * dt * k2v1;
        const double k3x2 = v2 + 0.5 * dt * k2v2;
        double k3v0, k3v1, k3v2;
        acceleration_2d(x0 + 0.5 * dt * k2x0, x1 + 0.5 * dt * k2x1, x2 + 0.5 * dt * k2x2, qmk, fx, fy,
                        origin_x, origin_y, h, cylindrical, &k3v0, &k3v1, &k3v2);
        const double k4x0 = v0 + dt * k3v0;
        const double k4x1 = v1 + dt * k3v1;
        const double k4x2 = v2 + dt * k3v2;
        double k4v0, k4v1, k4v2;
        acceleration_2d(x0 + dt * k3x0, x1 + dt * k3x1, x2 + dt * k3x2, qmk, fx, fy,
                        origin_x, origin_y, h, cylindrical, &k4v0, &k4v1, &k4v2);
        y0 = x0 + dt / 6.0 * (v0 + 2.0 * k2x0 + 2.0 * k3x0 + k4x0);
        y1 = x1 + dt / 6.0 * (v1 + 2.0 * k2x1 + 2.0 * k3x1 + k4x1);
        y2 = x2 + dt / 6.0 * (v2 + 2.0 * k2x2 + 2.0 * k3x2 + k4x2);
        w0 = v0 + dt / 6.0 * (a0 + 2.0 * k2v0 + 2.0 * k3v0 + k4v0);
        w1 = v1 + dt / 6.0 * (a1 + 2.0 * k2v1 + 2.0 * k3v1 + k4v1);
        w2 = v2 + dt / 6.0 * (a2 + 2.0 * k2v2 + 2.0 * k3v2 + k4v2);
        }
        const double m0 = 0.5 * (x0 + y0) + dt * (v0 - w0) / 8.0;
        const double m1 = 0.5 * (x1 + y1) + dt * (v1 - w1) / 8.0;
        const double m2 = 0.5 * (x2 + y2) + dt * (v2 - w2) / 8.0;
        const double my = cylindrical != 0 ? hypot(m1, m2) : m1;
        deposit_point_2d(m0, my, dqk * dt, origin_x, origin_y, h, cylindrical, charge);
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
                    hits(k, p, c, 3) = v0 + s * (w0 - v0);
                    hits(k, p, c, 4) = v1 + s * (w1 - v1);
                    hits(k, p, c, 5) = v2 + s * (w2 - v2);
                    hits(k, p, c, 6) = after >= 0.0 ? 1.0 : -1.0;
                    hits(k, p, c, 7) = 1.0 * ns;
                }
                counts(k, p) = c + 1;
            }
        }
        x0 = y0; x1 = y1; x2 = y2;
        v0 = w0; v1 = w1; v2 = w2;
        const double px = x0;
        const double py = cylindrical != 0 ? hypot(x1, x2) : x1;
        if (slot >= 0) {
            const long long c = path_count(slot);
            if (c < path_capacity) {
                paths(slot, c, 0) = px;
                paths(slot, c, 1) = py;
            }
            path_count(slot) = c + 1;
        }
        if (px >= upper_x) {
            st = 0;
        } else if (px < origin_x) {
            st = 1;
        } else if (py > upper_y || (cylindrical == 0 && py < origin_y)) {
            st = 2;
        } else {
            const long long i = min(max((long long)((px - origin_x) / h), 0LL), nx - 2);
            const long long j = min(max((long long)((py - origin_y) / h), 0LL), ny - 2);
            const long long cell = cells(i, j);
            if (cell >= 0) {
                const long long si = min(max((long long)(((px - origin_x) / h - i) * refine), 0LL), refine - 1);
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
