// Exercise 21 - int4 dequant GEMV
//
//   y[M] = W[M, K] @ x[K]
//
// W is stored as 4-bit integers, grouped along K with GROUP = 128, one fp16
// scale and one fp16 zero point per group:
//
//   w[m][k] = (q[m][k] - zeros[m][k/GROUP]) * scales[m][k/GROUP]
//
// The packing the harness hands you is the naive one: element k lives in
// nibble (k % 8) of word (k / 8), rows contiguous, K/8 words per row.
//
// This kernel is a memcpy with arithmetic hidden in its shadow. The gate is
// 80% of your measured copy ceiling, counting packed weights, scales, zeros,
// x and y. TFLOP/s is not a meaningful number here and the harness will not
// print it.

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GROUP 128

// One-time weight transformation, NOT timed -- in production this happens at
// model load. The identity is provided so the exercise runs; replacing it with
// a layout that makes your unpack cheaper is chapter 21 exercise 21.3, and is
// where a large part of the remaining performance is.
//
// If you change this, your kernel must read the layout you produced. The
// reference always reads the original.
__global__ void repack_identity(const uint32_t* __restrict__ src,
                                uint32_t* __restrict__ dst, size_t words) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) dst[i] = src[i];
}

void launch_repack(const uint32_t* src, uint32_t* dst, int M, int K) {
    size_t words = size_t(M) * (K / 8);
    repack_identity<<<int((words + 255) / 256), 256>>>(src, dst, words);
}

__global__ void gemv_int4(const uint32_t* __restrict__ qweight,
                          const half* __restrict__ scales,
                          const half* __restrict__ zeros,
                          const half* __restrict__ x, float* __restrict__ y,
                          int M, int K) {
    // TODO.
    //
    // Before writing: how many bytes must this kernel read, and how long does
    // that take at your copy ceiling? That is your target, and nothing in the
    // inner loop can beat it.
    //
    // Then, in order of how much they matter:
    //   - how wide is each thread's load of qweight?
    //   - where does x live? (it is K halves, read by every row)
    //   - how many rows per block, and how many blocks? is the GPU full?
    //   - how often do you load a scale, versus how often do you use it?
    //   - what is the final reduction, and is it touching shared memory more
    //     than it needs to?
}

void launch_gemv(const uint32_t* qweight, const half* scales, const half* zeros,
                 const half* x, float* y, int M, int K) {
    // TODO: choose the grid and block shape.
    (void)qweight; (void)scales; (void)zeros; (void)x; (void)y; (void)M; (void)K;
}
