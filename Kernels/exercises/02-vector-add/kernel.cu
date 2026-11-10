// Exercise 02 - vector add
//
//   c[i] = a[i] + b[i]   for i in [0, n)
//
// The simplest possible memory-bound kernel. Your job is to make it reach the
// memory roof, which is the only thing a kernel this trivial can do.
//
// You own both the kernel body and the launch configuration. Chapter 02 gives
// you the index arithmetic; Chapter 01 tells you what number to aim at.

#include <cuda_runtime.h>

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    // TODO: compute this thread's global index, guard it, and do the add.
    //
    // Then, once it passes: look at the launch below. Is one element per
    // thread the right amount of work? Chapter 01 section 4, and Little's Law
    // in Chapter 00, both have opinions about bytes in flight per thread.
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid = 1;  // TODO
    vector_add<<<grid, block>>>(a, b, c, n);
}
