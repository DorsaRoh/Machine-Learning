// Exercise 13 - tensor core GEMM
//
//   C = A @ B,  A and B are half (fp16), C is float.  Row-major.
//
// Accumulation in fp32 is mandatory: with K = 4096 an fp16 accumulator loses
// the sum entirely, and the correctness gate will tell you so.
//
// Two versions:
//   v1  WMMA        nvcuda::wmma fragments. The approachable API.
//   v2  mma.sync    inline PTX with ldmatrix and a swizzled shared layout.
//
// v1 alone can clear the 50%-of-cuBLAS bar on most GPUs. v2 is the stretch,
// and it is the one that teaches you what WMMA was hiding.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define IMPLEMENTED 1  // TODO: 1 or 2

// ------------------------------------------------------------------ v1 WMMA
__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    // Structure:
    //   - each warp owns some tile of C, held in accumulator fragments
    //   - loop over K in steps of 16
    //       stage A and B tiles into shared memory (coalesced, 16-byte aligned)
    //       __syncthreads()
    //       load_matrix_sync into a_frag / b_frag from SHARED, not global
    //       mma_sync
    //   - store_matrix_sync at the end
    //
    // Rules that will bite you:
    //   - every _sync call must be reached by all 32 lanes of the warp
    //   - you may NOT index into a fragment; its layout is opaque
    //   - leading dimensions must be multiples of 8 for fp16
    //   - loading fragments straight from global works and is slow. Stage in
    //     shared and reuse, exactly as in Chapter 11.
    //
    // Start by getting ONE 16x16x16 tile right, with M=N=K=16, before building
    // the loop.
}

// --------------------------------------------------------------- v2 mma.sync
__global__ void gemm_mma(const half* __restrict__ A, const half* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
    // TODO (stretch).
    //
    //   - shared tiles with an XOR swizzle (padding breaks ldmatrix alignment)
    //   - ldmatrix.sync.aligned.m8n8.x4.shared.b16 to build fragments
    //   - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    //   - the register layout is specified in the PTX ISA docs. You will need
    //     the diagrams open. Everyone does.
    //
    // Debugging advice: write a tiny kernel that fills A with the value of its
    // own linear index, runs one ldmatrix, and prints what each lane holds.
    // Compare against the documented layout. Do that before building anything.
}

void launch_tc_gemm(int version, const half* A, const half* B, float* C, int M,
                    int N, int K) {
    // TODO
    (void)version; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}
