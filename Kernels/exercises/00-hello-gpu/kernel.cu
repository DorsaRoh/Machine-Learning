// Exercise 00 - hello-gpu
//
// Goal: confirm your toolchain works, and write the single most important line
// in CUDA.
//
// Every thread in the grid must compute its own unique index into a flat
// array. There are `gridDim.x` blocks, each with `blockDim.x` threads. Within
// a block, a thread knows itself as `threadIdx.x`; the block knows itself as
// `blockIdx.x`.
//
// Fill in `global_index()` so that across the whole grid, every thread gets a
// distinct value in [0, gridDim.x * blockDim.x).
//
// Then write out[i] = i for every valid i. `n` may not be a multiple of the
// block size, so some threads in the last block have no work. Decide what they
// should do.

#include <cuda_runtime.h>

__device__ __forceinline__ int global_index() {
    // TODO: replace this. threadIdx.x alone repeats in every block.
    return threadIdx.x;
}

__global__ void write_indices(int* out, int n) {
    int i = global_index();

    // TODO: guard against i >= n before writing, then write out[i] = i.
    out[i] = i;
}

// Launch configuration. You choose. The harness calls this.
void launch_write_indices(int* d_out, int n) {
    int block = 256;
    int grid = 1;  // TODO: how many blocks do you need to cover n elements?
    write_indices<<<grid, block>>>(d_out, n);
}
