// Exercise 20 - FlashAttention backward
//
// Inputs, all [B*H, N, d] row-major unless noted:
//   Q, K, V, O, dO : fp16
//   L              : fp32, [B*H, N], the logsumexp saved by the forward
// Outputs:
//   dQ, dK, dV     : fp32, [B*H, N, d]
//
// The recurrence (chapter 20 section 4), for a tile pair (i, j):
//
//   S_ij  = Q_i K_j^T * scale
//   P_ij  = exp(S_ij - L_i)          <- already normalized; this is why L
//   dV_j += P_ij^T dO_i
//   dP_ij = dO_i V_j^T
//   D_i   = rowsum(dO_i * O_i)       <- precompute, own kernel
//   dS_ij = P_ij * (dP_ij - D_i)
//   dQ_i += dS_ij K_j * scale
//   dK_j += dS_ij^T Q_i * scale
//
// The workspace is O(B*H*N*(d+8)) floats: enough for D, and enough for one
// fp32 dQ accumulator, and not enough for anything [N, N].
//
// Gate: dQ, dK, dV within 3e-3 of the unfused reference; 2.0x over it.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BR 64
#define BC 64

// D_i = rowsum(dO_i * O_i). One of the three loop orders in section 4 needs
// this before the main loop; all three need it somewhere.
__global__ void compute_D(const half* __restrict__ dO, const half* __restrict__ O,
                          float* __restrict__ D, int N, int d) {
    // TODO
}

__global__ void flash_bwd(const half* __restrict__ Q, const half* __restrict__ K,
                          const half* __restrict__ V, const half* __restrict__ dO,
                          const float* __restrict__ L, const float* __restrict__ D,
                          float* __restrict__ dQ, float* __restrict__ dK,
                          float* __restrict__ dV, int N, int d, float scale,
                          int causal) {
    // TODO.
    //
    // Decide the loop order first, and write down why (chapter 20, 20.4):
    //   - j outer, dQ via atomicAdd          : one kernel, non-deterministic
    //   - two kernels, one per loop order    : deterministic, recomputes S, P
    //   - j outer, fp32 dQ accumulator split : deterministic, more complex
    //
    // Whichever you pick, the thing that must stay in registers across the
    // inner loop is whatever the OUTER index owns.
}

void launch_flash_bwd(const half* Q, const half* K, const half* V, const half* O,
                      const half* dO, const float* L, float* dQ, float* dK,
                      float* dV, int BH, int N, int d, float scale, bool causal,
                      float* workspace, size_t workspace_floats) {
    // TODO: zero whatever you accumulate into, run compute_D, then the main
    // kernel (or kernels).
    (void)Q; (void)K; (void)V; (void)O; (void)dO; (void)L;
    (void)dQ; (void)dK; (void)dV; (void)BH; (void)N; (void)d;
    (void)scale; (void)causal; (void)workspace; (void)workspace_floats;
}
