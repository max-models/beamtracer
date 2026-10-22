// The CUDA version (GPU) of the staggered_field_2d kernel; staggered_field_2d_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of points: one thread per point.

#include "cunumpy/array_view.cuh"

__device__ __forceinline__ double beamtracer_clamped_bilinear(
    const Array2D<double>& f, double s, double t, long long imax, long long jmax) {
    const long long i = min(max((long long)floor(s), 0LL), imax);
    const long long j = min(max((long long)floor(t), 0LL), jmax);
    const double u = s - i;
    const double w = t - j;
    const double low = f(i, j) + w * (f(i, j + 1) - f(i, j));
    const double high = f(i + 1, j) + w * (f(i + 1, j + 1) - f(i + 1, j));
    return low + u * (high - low);
}

extern "C" __global__
void staggered_field_2d(
    Array1D<double> px,
    Array1D<double> py,
    Array2D<double> fx,
    Array2D<double> fy,
    double origin_x,
    double origin_y,
    double h,
    Array1D<double> ex,
    Array1D<double> ey
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= px.shape[0]) return;
    const long long nx = fy.shape[0];
    const long long ny = fx.shape[1];
    const double sx = (px(k) - origin_x) / h;
    const double sy = (py(k) - origin_y) / h;
    ex(k) = beamtracer_clamped_bilinear(fx, sx + 0.5, sy, nx - 1, ny - 2);
    ey(k) = beamtracer_clamped_bilinear(fy, sx, sy + 0.5, nx - 2, ny - 1);
}
