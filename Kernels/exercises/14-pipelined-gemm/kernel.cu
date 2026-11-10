// Exercise 14 - pipelined GEMM
//
// Take your own best GEMM and make the global->shared path asynchronous.
// The gate is a SPEEDUP OVER YOUR OWN BASELINE, not a fraction of cuBLAS,
// because the thing being measured here is the pipelining, not the kernel.
//
//   baseline  : paste your 11-gemm-ladder v5 (or 13's WMMA kernel) here
//   pipelined : same tiles, same math, asynchronous multi-stage loads
//
// On sm_80+ use cp.async / __pipeline_memcpy_async with 2-5 stages.
// On older hardware (T4, sm_75) cp.async does not exist; build the
// register-staged double buffer from Chapter 12 section 2 instead. The gate is
// the same and so is the idea.

#include <cuda_runtime.h>

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
#include <cuda_pipeline.h>
#endif

#define STAGES 3  // TODO: sweep this. 2, 3, 4, 5.

// ------------------------------------------------------------------ baseline
__global__ void gemm_baseline(const float* __restrict__ A,
                              const float* __restrict__ B, float* __restrict__ C,
                              int M, int N, int K) {
    // TODO: paste your best synchronous GEMM here, unchanged.
    //
    // It must be YOUR kernel. The gate compares against it, so a weak baseline
    // makes the bar trivial and teaches you nothing.
}

// ----------------------------------------------------------------- pipelined
__global__ void gemm_pipelined(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
    // TODO.
    //
    // Structure (sm_80+):
    //
    //   __shared__ float As[STAGES][BK][BM], Bs[STAGES][BK][BN];
    //
    //   prologue: for s in 0..STAGES-2: issue copies for tile s, commit
    //   loop k:
    //       __pipeline_wait_prior(STAGES - 2);
    //       __syncthreads();
    //       issue copies for tile k + STAGES - 1 into stage (k+STAGES-1)%STAGES
    //       __pipeline_commit();
    //       compute on stage k % STAGES
    //
    // Things that will go wrong, in the order they will go wrong:
    //   1. forgetting that cp.async completion is NOT implied by
    //      __syncthreads() -- you need the wait
    //   2. off-by-one in which stage you compute on versus which you load
    //   3. the tail: the last STAGES-1 iterations have nothing left to issue
    //   4. cp.async requires 4, 8 or 16 byte sizes with matching alignment,
    //      and silently does nothing useful if you get it wrong
    //
    // Verify with: make sass | grep LDGSTS
    // No LDGSTS means no async copy happened.
}

void launch_gemm(int version, const float* A, const float* B, float* C, int M,
                 int N, int K) {
    // version 0 = baseline, 1 = pipelined
    (void)version; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}
