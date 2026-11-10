// Exercise 03 - naive transpose
//
//   out[j * n + i] = in[i * n + j]
//
// You are NOT asked to make this fast. You are asked to make it correct, then
// measure the damage and account for it to within 20%.
//
// Write the obvious version. One thread per output element, 2D indexing, no
// shared memory. That is the point: the obvious version is the one whose cost
// you need to understand before Chapter 06 shows you the way out.

#include <cuda_runtime.h>

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    // TODO: 2D indexing, bounds guard, the transposing write.
    //
    // Before you write it, decide which of threadIdx.x and threadIdx.y should
    // map to the row and which to the column. You have two choices and they
    // are not equivalent. Pick deliberately, then try the other one and
    // measure the difference.
}

void launch_transpose(const float* in, float* out, int n) {
    dim3 block(32, 8);
    dim3 grid(1, 1);  // TODO
    transpose_naive<<<grid, block>>>(in, out, n);
}
