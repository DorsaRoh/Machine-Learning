// Exercise 22 - MoE dispatch
//
//   x           [T, H]        fp16, the tokens
//   W           [E, H, H]     fp16, one [H, H] matrix per expert, row-major
//   expert_ids  [T, KTOP]     int, which experts this token goes to
//   probs       [T, KTOP]     float, the router weight for each assignment
//   y           [T, H]        fp32, output
//
//   y[t] = sum over slots s of  probs[t][s] * (x[t] @ W[expert_ids[t][s]])
//
// The dense-masked reference runs every token through every expert: E/KTOP = 4x
// the necessary work. Your job is to do only the necessary work and keep at
// least 70% of that theoretical 4x, which means >= 2.8x over the reference.
//
// The pipeline (chapter 22 section 4):
//
//   1. histogram expert_ids           -> counts[E]          (chapter 09)
//   2. exclusive scan counts          -> offsets[E]         (chapter 08)
//   3. scatter assignment indices into permuted order
//   4. gather token rows into a permuted buffer
//   5. one GEMM per expert over its contiguous row range
//   6. scatter results back, weighted by probs, summed over slots
//
// Every step except 5 is something you have already built.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Steps 1-3. `perm` should end up listing assignment indices (0 .. T*KTOP-1)
// grouped by expert; `offsets` the start of each expert's range.
__global__ void build_permutation(const int* __restrict__ expert_ids,
                                  int* __restrict__ counts, int* __restrict__ offsets,
                                  int* __restrict__ perm, int T, int ktop, int E) {
    // TODO. You may split this into several kernels; it is one declaration
    // here only so the file starts compiling.
}

// Step 4.
__global__ void gather_rows(const half* __restrict__ x, const int* __restrict__ perm,
                            half* __restrict__ xp, int H, int ktop, int total) {
    // TODO. Before writing: how many contiguous bytes does each index cover?
    // That number decides whether this gather costs anything at all.
}

// Step 6.
__global__ void scatter_combine(const float* __restrict__ yp,
                                const int* __restrict__ perm,
                                const float* __restrict__ probs,
                                float* __restrict__ y, int H, int ktop, int total) {
    // TODO. Two slots per token land in the same output row. Think about
    // whether that needs an atomic, and whether you can arrange for it not to.
}

void launch_moe(const half* x, const half* W, const int* expert_ids,
                const float* probs, float* y, int T, int H, int E, int ktop,
                void* workspace, size_t workspace_bytes, cublasHandle_t handle) {
    // TODO.
    //
    // Step 5 is E GEMMs with different row counts. Two honest options:
    //   - a loop of cublasGemmEx, which needs the counts on the HOST, which
    //     means a cudaMemcpy and a synchronization. Measure what that costs.
    //   - your own grouped GEMM: one grid, each block binary-searches offsets
    //     to find which expert and which tile it owns. No sync.
    //
    // The second is strictly better and strictly harder. Do the first, measure,
    // then decide whether the sync is worth removing.
    (void)x; (void)W; (void)expert_ids; (void)probs; (void)y;
    (void)T; (void)H; (void)E; (void)ktop;
    (void)workspace; (void)workspace_bytes; (void)handle;
}
