// Exercise 03 - the stride study
//
//   out[i] = in[i * stride]      for i in [0, n)
//
// One kernel, swept across strides. The point is not to make it fast -- at
// stride 32 nothing can be fast. The point is to produce the coalescing
// penalty curve for your GPU and be able to explain every feature of it.
//
// Write the kernel so that the ONLY thing varying across the sweep is the
// access pattern. If your implementation also changes the amount of work, the
// grid shape, or the number of instructions as stride changes, the curve
// measures your kernel instead of the memory system.

#include <cuda_runtime.h>

__global__ void strided_read(const float* __restrict__ in,
                             float* __restrict__ out, int n, int stride) {
    // TODO: global index, bounds guard, strided read, contiguous write.
    //
    // Note the asymmetry: the READ is strided and the WRITE is contiguous.
    // Keep it that way. You want to isolate one effect.
    //
    // Watch your index types. n * stride can exceed 2^31.
}

void launch_strided_read(const float* in, float* out, int n, int stride) {
    int block = 256;
    int grid = 1;  // TODO
    strided_read<<<grid, block>>>(in, out, n, stride);
}
