// Exercise 11 - the GEMM ladder
//
//   C = A @ B,  row-major,  A: M x K,  B: K x N,  C: M x N,  fp32
//
// Five versions. The harness runs every version up to IMPLEMENTED, checks each
// against cuBLAS, and reports TFLOP/s and % of cuBLAS for each.
//
//   v1  naive                 one thread per output, loop over K
//   v2  shared tiling         BM x BN x BK tile in shared memory
//   v3  1D register tiling    each thread computes TM outputs
//   v4  2D register tiling    each thread computes TM x TN outputs
//   v5  vectorized + tuned    float4, transposed A tile, no bank conflicts
//
// Chapter 11 gives the shape of each step and the question it answers.
// Chapter 12 gives the list of things standing between v5 and cuBLAS.
// Neither gives code.
//
// Before each version, fill in the prediction block in the README. That is not
// ceremony -- a version whose speedup you cannot predict is a version you do
// not understand.

#include <cuda_runtime.h>

#define IMPLEMENTED 1  // TODO: raise as you climb

// ------------------------------------------------------------------ v1
__global__ void gemm_v1(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    // One decision before you write a line: which of threadIdx.x and
    // threadIdx.y maps to the row of C, and which to the column? The two
    // choices differ by about 8x. Chapter 03 if you need reminding why.
}

// ------------------------------------------------------------------ v2
// Shared-memory tiling. Suggested start: BM = BN = BK = 32, 32x32 threads.
__global__ void gemm_v2(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    //   for kt in 0..K/BK:
    //       cooperatively load A tile and B tile into shared
    //       __syncthreads()
    //       accumulate from shared
    //       __syncthreads()     <- do not skip this one
    //
    // Both tile loads must be coalesced. Check by tracing warp 0.
}

// ------------------------------------------------------------------ v3
// 1D register tiling: each thread computes TM outputs in a column.
__global__ void gemm_v3(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    // The TM accumulators must live in registers. That means the loop over TM
    // must be fully unrolled with compile-time bounds -- otherwise the array
    // is dynamically indexed and spills to local memory (Chapter 04).
    // Verify with: nvcc -Xptxas -v  (look for "spill")
}

// ------------------------------------------------------------------ v4
// 2D register tiling: each thread computes a TM x TN block. The inner loop is
// an outer product.
__global__ void gemm_v4(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    // Suggested: BM=BN=128, BK=8, TM=TN=8, 256 threads.
    // Before writing: how many registers? What occupancy? How many FMAs per
    // shared-memory load? (Chapter 11 exercise 11.2.)
}

// ------------------------------------------------------------------ v5
// Vectorized loads, transposed A tile, conflict-free shared layout, tuned.
__global__ void gemm_v5(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    // The list, from Chapter 12:
    //   - float4 global loads for both tiles
    //   - store the A tile TRANSPOSED in shared, so the inner-loop column read
    //     becomes contiguous and float4-able
    //   - float4 shared loads in the inner loop
    //   - check bank conflicts with ncu; pad or swizzle as needed
    //   - sweep BM/BN/BK/TM/TN
}

// ------------------------------------------------------------------ dispatch
void launch_gemm(int version, const float* A, const float* B, float* C, int M,
                 int N, int K) {
    // TODO: grid/block per version.
    (void)version; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}
