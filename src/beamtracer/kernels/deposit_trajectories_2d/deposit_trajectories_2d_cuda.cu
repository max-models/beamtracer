// The CUDA version (GPU) of the deposit_trajectories_2d kernel; deposit_trajectories_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of charges: one thread per charge, adding atomically.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"

extern "C" __global__
void deposit_trajectories_2d(
    Array1D<double> px,
    Array1D<double> py,
    Array1D<double> q,
    double origin_x,
    double origin_y,
    double h,
    long long cylindrical,
    Array2D<double> charge
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= px.shape[0]) return;
    const long long nx = charge.shape[0];
    const long long ny = charge.shape[1];
    const double x = px(k);
    const double y = py(k);
    long long i = (long long)floor((x - origin_x) / h);
    long long j = (long long)floor((y - origin_y) / h);
    double t = (x - (origin_x + h * i)) / h;
    double u;
    if (cylindrical != 0) {
        const double r1 = origin_y + h * j;
        const double r2 = r1 + h;
        u = (y * y - r1 * r1) / (r2 * r2 - r1 * r1);
    } else {
        u = (y - (origin_y + h * j)) / h;
    }
    if (i < 0) {
        t = 0.0;
        i = 0;
    } else if (i >= nx - 1) {
        t = 1.0;
        i = nx - 2;
    }
    if (j < 0) {
        u = 0.0;
        j = 0;
    } else if (j >= ny - 1) {
        u = 1.0;
        j = ny - 2;
    }
    const double value = q(k);
    cunumpy_atomic_add(&charge(i, j), (1.0 - t) * (1.0 - u) * value);
    cunumpy_atomic_add(&charge(i, j + 1), (1.0 - t) * u * value);
    cunumpy_atomic_add(&charge(i + 1, j), t * (1.0 - u) * value);
    cunumpy_atomic_add(&charge(i + 1, j + 1), t * u * value);
}
