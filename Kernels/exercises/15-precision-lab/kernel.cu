// Exercise 15 - the precision lab
//
// No performance gate. The deliverable is a written explanation of every
// number this produces.
//
// Four experiments, each isolating one numerical phenomenon:
//
//   E1  accumulation order    : sum n values four ways, compare to float64
//   E2  cancellation          : variance via the identity vs Welford
//   E3  accumulator width     : dot product with fp32 / fp16 accumulators
//   E4  format range          : what overflows in fp16 but not bf16
//
// You implement the kernels. The harness computes the float64 ground truth on
// the CPU and prints the error table.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// --------------------------------------------------------------- E1: sums
// Four reductions of the same data. All four must produce a scalar in out[0].
//   mode 0: sequential   -- ONE thread sums everything, in order. Slow. The
//                           point is the error, not the time.
//   mode 1: tree         -- your reduction from 07-reduce
//   mode 2: Kahan        -- compensated summation, in the per-thread partial
//   mode 3: fp64 accum   -- same tree, double accumulators
__global__ void sum_kernel(const float* __restrict__ in, double* __restrict__ out,
                           size_t n, int mode) {
    // TODO
}

// -------------------------------------------------------- E2: variance
// Compute the variance of each row two ways.
//   mode 0: var = E[x^2] - E[x]^2, fp32 accumulation
//   mode 1: Welford, fp32
//   mode 2: var = E[x^2] - E[x]^2, fp64 accumulation
__global__ void variance_kernel(const float* __restrict__ x,
                                float* __restrict__ var, int rows, int cols,
                                int mode) {
    // TODO
}

// ---------------------------------------------------------- E3: dot product
// out[0] = sum_i a[i] * b[i], with a and b in fp16.
//   mode 0: fp32 accumulator
//   mode 1: fp16 accumulator   <- watch this one fail as n grows
__global__ void dot_half(const half* __restrict__ a, const half* __restrict__ b,
                         float* __restrict__ out, int n, int mode) {
    // TODO
}

// --------------------------------------------------------------- E4: range
// y[i] = x[i] * x[i], computed and stored in the given format, then widened
// back to float. The harness feeds values that straddle each format's limits.
//   mode 0: fp32   mode 1: fp16   mode 2: bf16
__global__ void square_in_format(const float* __restrict__ x,
                                 float* __restrict__ y, int n, int mode) {
    // TODO
}

void launch_sum(const float* in, double* out, size_t n, int mode);
void launch_variance(const float* x, float* var, int rows, int cols, int mode);
void launch_dot(const half* a, const half* b, float* out, int n, int mode);
void launch_square(const float* x, float* y, int n, int mode);

// TODO: define the four launchers.
