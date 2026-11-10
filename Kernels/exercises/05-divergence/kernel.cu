// Exercise 05 - the cost of divergence
//
// Each thread runs `work` steps of a linear congruential generator. Which
// constants it uses depends on a branch controlled by its thread index and a
// `divisor` the harness sweeps:
//
//     take_a = ((tid % divisor) * 2 < divisor)
//
//   divisor = 1   -> every thread takes B. No warp diverges.
//   divisor = 2   -> alternating lanes. Every warp diverges maximally.
//   divisor = 64  -> whole warps agree again (32 lanes of A, then 32 of B).
//
// Integer arithmetic, so correctness is exact and the compiler cannot fold
// anything away.
//
// Write it so the ONLY difference between configurations is which lanes take
// which path. Equal work on both sides.

#include <cstdint>

#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t lcg_a(uint32_t v, int steps) {
    for (int i = 0; i < steps; ++i) v = v * 1664525u + 1013904223u;
    return v;
}

__device__ __forceinline__ uint32_t lcg_b(uint32_t v, int steps) {
    for (int i = 0; i < steps; ++i) v = v * 22695477u + 1u;
    return v;
}

__global__ void divergent(uint32_t* __restrict__ out, int n, int divisor,
                          int work) {
    // TODO:
    //   1. global index, bounds guard
    //   2. seed:  uint32_t v = (uint32_t)i;
    //   3. branch on ((i % divisor) * 2 < divisor): lcg_a else lcg_b
    //   4. out[i] = v;
    //
    // The CPU reference in main.cu does exactly this, so any deviation shows
    // up as a correctness failure rather than a mysterious speedup.
}

void launch_divergent(uint32_t* out, int n, int divisor, int work) {
    int block = 256;
    int grid = 1;  // TODO
    divergent<<<grid, block>>>(out, n, divisor, work);
}
