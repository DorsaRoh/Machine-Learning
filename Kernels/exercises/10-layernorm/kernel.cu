// Exercise 10 - LayerNorm
//
//   mean_r = (1/D) sum_c x[r][c]
//   var_r  = (1/D) sum_c (x[r][c] - mean_r)^2
//   y[r][c] = gamma[c] * (x[r][c] - mean_r) * rsqrt(var_r + eps) + beta[c]
//
// Two gates, and the second one is the interesting one:
//
//   1. Performance: 85% of copy ceiling on well-conditioned input.
//   2. Stability:   the harness also runs x ~ N(1000, 1), where the naive
//                   E[x^2] - E[x]^2 identity loses all its significant digits
//                   and can produce a negative variance. Your kernel must
//                   survive that.
//
// You can pass gate 1 with the naive identity. You cannot pass gate 2 with it.

#include <cuda_runtime.h>

// TODO: your kernel(s).
//
// Decisions:
//   - one pass or two? (gamma and beta are tiny and reused by every row --
//     where should they live?)
//   - naive sum/sumsq identity, or Welford? Chapter 10 section 3.
//   - Welford's merge formula is associative, so it fits the same block
//     reduction you built in 07-reduce. What does the carried state look like,
//     and how many floats is it per lane?
//   - accumulate in float or double? What does double cost on your GPU?
//     (Check the fp64 rate for your chip -- on consumer cards it is 1/64 of
//     fp32 and the answer may surprise you.)

void launch_layernorm(const float* x, const float* gamma, const float* beta,
                      float* y, int rows, int cols, float eps) {
    // TODO
    (void)x; (void)gamma; (void)beta; (void)y; (void)rows; (void)cols; (void)eps;
}
