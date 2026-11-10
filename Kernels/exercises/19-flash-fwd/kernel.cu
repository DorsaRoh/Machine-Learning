// Exercise 19 - FlashAttention forward
//
// Layout: Q, K, V are [B*H, N, d] in row-major, contiguous, fp16.
//         O is      [B*H, N, d] fp32.
//         L is      [B*H, N]    fp32, the per-row logsumexp:  L = m + log(l).
//
// L is not optional. Exercise 20 needs it, and the harness checks it.
//
// The workspace you are given is O(B*H*N) floats, NOT O(B*H*N*N). This is
// deliberate: an implementation that materializes S will not fit, and the
// harness will tell you so rather than letting you discover it at N = 16384.
//
// Gate: 2.5x over the three-kernel cuBLAS baseline, O within 2e-3, L within
// 2e-3.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Suggested starting point. Both are yours to change; changing them is most of
// the exercise. Budget your registers first (chapter 19 exercise 19.1) -- do
// not pick these by trial and error.
#define BR 64   // query rows per block
#define BC 64   // key/value rows per tile

__global__ void flash_fwd(const half* __restrict__ Q, const half* __restrict__ K,
                          const half* __restrict__ V, float* __restrict__ O,
                          float* __restrict__ L, int N, int d, float scale,
                          int causal) {
    // TODO.
    //
    // The recurrence, from chapter 19 section 4:
    //
    //   m_new = max(m, rowmax(S_j))
    //   corr  = exp(m - m_new)
    //   P_j   = exp(S_j - m_new)
    //   l     = l * corr + rowsum(P_j)
    //   O     = O * corr + P_j @ V_j
    //   m     = m_new
    //
    // and once, at the end:  O /= l,  L = m + log(l).
    //
    // Things to decide before you type:
    //   - which index does blockIdx.x carry, and which blockIdx.y?
    //   - where does O live for the duration of the loop, and how many
    //     registers per thread is that?
    //   - rowmax and rowsum are reductions across which threads, in your
    //     chosen layout? (chapter 13 section 4 if you are using mma fragments)
    //   - does S_j ever need to touch shared memory?
    //
    // Start with a plain fp32 version that is correct, confirm it against the
    // baseline, and only then bring in wmma or mma.sync. A fast wrong kernel
    // is worth nothing and is much harder to debug.
}

// The harness calls this. Choose your own grid and block shape.
void launch_flash_fwd(const half* Q, const half* K, const half* V, float* O,
                      float* L, int BH, int N, int d, float scale, bool causal,
                      float* workspace, size_t workspace_floats) {
    // TODO: launch flash_fwd over all BH (batch * head) sequences.
    //
    // You probably do not need the workspace at all. It is here so that you
    // can tell: if you reach for it and it is too small, your algorithm is
    // still O(N^2) in memory.
    (void)Q; (void)K; (void)V; (void)O; (void)L; (void)BH; (void)N; (void)d;
    (void)scale; (void)causal; (void)workspace; (void)workspace_floats;
}
