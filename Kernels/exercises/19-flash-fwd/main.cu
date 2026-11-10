// Harness for 19-flash-fwd. Do not edit.
//
// The baseline is the honest three-kernel version: cuBLAS QK^T, a softmax
// kernel, cuBLAS PV. It is also the correctness oracle, computed with the same
// fp16 inputs and fp32 accumulation your kernel should be using, so any
// disagreement is your kernel and not the dtype.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

#define CUBLAS_CHECK(expr)                                                 \
    do {                                                                   \
        cublasStatus_t st_ = (expr);                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(st_),   \
                         __FILE__, __LINE__);                              \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// One block per row of S. Reads S (fp32), writes P (fp16) and L (fp32).
__global__ void ref_softmax(const float* __restrict__ S, half* __restrict__ P,
                            float* __restrict__ L, int N, int causal) {
    int row = blockIdx.x;
    int i = row % N;  // position within its own sequence
    const float* s = S + size_t(row) * N;
    half* p = P + size_t(row) * N;
    int limit = causal ? (i + 1) : N;

    __shared__ float red[32];
    __shared__ float out_m, out_l;
    int nwarps = (blockDim.x + 31) / 32;

    float m = -INFINITY;
    for (int j = threadIdx.x; j < limit; j += blockDim.x) m = fmaxf(m, s[j]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (int(threadIdx.x) < nwarps) ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (threadIdx.x == 0) out_m = v;
    }
    __syncthreads();
    m = out_m;
    __syncthreads();

    float l = 0.0f;
    for (int j = threadIdx.x; j < limit; j += blockDim.x) l += __expf(s[j] - m);
    for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = l;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (int(threadIdx.x) < nwarps) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) out_l = v;
    }
    __syncthreads();
    l = out_l;

    float inv = 1.0f / l;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        p[j] = __float2half(j < limit ? __expf(s[j] - m) * inv : 0.0f);
    if (threadIdx.x == 0) L[row] = m + logf(l);
}

struct Baseline {
    cublasHandle_t h;
    float* S;
    half* P;
    int BH, N, d;
    float scale;
    bool causal;

    void run(const half* Q, const half* K, const half* V, float* O, float* L) {
        const float one = 1.0f, zero = 0.0f;
        // S_row = Q K^T * scale, computed as column-major (K)(Q^T).
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &scale, K, CUDA_R_16F, d,
            (long long)N * d, Q, CUDA_R_16F, d, (long long)N * d, &zero, S,
            CUDA_R_32F, N, (long long)N * N, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        ref_softmax<<<BH * N, 256>>>(S, P, L, N, causal ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
        // O_row = P V, computed as column-major (V)(P).
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &one, V, CUDA_R_16F, d,
            (long long)N * d, P, CUDA_R_16F, N, (long long)N * N, &zero, O,
            CUDA_R_32F, d, (long long)N * d, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
};

int main(int argc, char** argv) {
    int B = (argc > 1) ? std::atoi(argv[1]) : 1;
    int H = (argc > 2) ? std::atoi(argv[2]) : 8;
    int N = (argc > 3) ? std::atoi(argv[3]) : 2048;
    int d = (argc > 4) ? std::atoi(argv[4]) : 64;
    bool causal = (argc > 5) ? std::atoi(argv[5]) != 0 : false;

    int BH = B * H;
    float scale = 1.0f / std::sqrt(float(d));
    bench::print_device();
    bench::Gate gate("19-flash-fwd");

    size_t qkv = size_t(BH) * N * d;
    std::vector<float> hq(qkv), hk(qkv), hv(qkv);
    bench::fill_random(hq, 11, -1.f, 1.f);
    bench::fill_random(hk, 22, -1.f, 1.f);
    bench::fill_random(hv, 33, -1.f, 1.f);
    std::vector<half> tmp(qkv);

    half *dQ, *dK, *dV, *dP;
    float *dO, *dL, *dOref, *dLref, *dS, *dWs;
    CUDA_CHECK(cudaMalloc(&dQ, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dO, qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOref, qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, size_t(BH) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLref, size_t(BH) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS, size_t(BH) * N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP, size_t(BH) * N * N * sizeof(half)));

    // O(N) workspace, on purpose. If your kernel needs more, it is not Flash.
    size_t ws_floats = size_t(BH) * N * 4;
    CUDA_CHECK(cudaMalloc(&dWs, ws_floats * sizeof(float)));

    for (size_t i = 0; i < qkv; ++i) tmp[i] = __float2half(hq[i]);
    CUDA_CHECK(cudaMemcpy(dQ, tmp.data(), qkv * sizeof(half), cudaMemcpyHostToDevice));
    for (size_t i = 0; i < qkv; ++i) tmp[i] = __float2half(hk[i]);
    CUDA_CHECK(cudaMemcpy(dK, tmp.data(), qkv * sizeof(half), cudaMemcpyHostToDevice));
    for (size_t i = 0; i < qkv; ++i) tmp[i] = __float2half(hv[i]);
    CUDA_CHECK(cudaMemcpy(dV, tmp.data(), qkv * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    Baseline base{handle, dS, dP, BH, N, d, scale, causal};

    base.run(dQ, dK, dV, dOref, dLref);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(dO, 0, qkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(dL, 0, size_t(BH) * N * sizeof(float)));
    launch_flash_fwd(dQ, dK, dV, dO, dL, BH, N, d, scale, causal, dWs, ws_floats);
    CUDA_CHECK_LAUNCH();

    std::vector<float> o(qkv), oref(qkv), l(size_t(BH) * N), lref(size_t(BH) * N);
    CUDA_CHECK(cudaMemcpy(o.data(), dO, qkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oref.data(), dOref, qkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(l.data(), dL, size_t(BH) * N * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lref.data(), dLref, size_t(BH) * N * 4, cudaMemcpyDeviceToHost));

    bench::ErrorStats eo = bench::compare(o, oref, 2e-3);
    bench::ErrorStats el = bench::compare(l, lref, 2e-3);
    std::printf("\nO: ");
    gate.correctness(eo, 2e-3);
    std::printf("L: max_abs %.3g, max_rel %.3g  %s\n", el.max_abs, el.max_rel,
                (el.first_bad == size_t(-1) && !el.nan_count) ? "ok"
                                                              : "WRONG (needed by ex 20)");

    double base_ms = bench::time_ms([&] { base.run(dQ, dK, dV, dOref, dLref); }, 3, 20, false);
    double flash_ms = bench::time_ms(
        [&] {
            launch_flash_fwd(dQ, dK, dV, dO, dL, BH, N, d, scale, causal, dWs,
                             ws_floats);
        },
        3, 20, false);

    double flops = 4.0 * double(BH) * N * N * d * (causal ? 0.5 : 1.0);
    double sq = double(BH) * N * N;
    std::printf("\nshape          : B=%d H=%d N=%d d=%d causal=%d\n", B, H, N, d,
                int(causal));
    std::printf("FLOPs          : %.2f GFLOP\n", flops / 1e9);
    std::printf("baseline       : %.3f ms  (%.1f TFLOP/s)\n", base_ms,
                bench::tflops(flops, base_ms));
    std::printf("yours          : %.3f ms  (%.1f TFLOP/s)\n", flash_ms,
                bench::tflops(flops, flash_ms));
    std::printf("S+P the baseline materializes : %.0f MB\n", sq * 6 / 1e6);
    std::printf("workspace you were given      : %.1f MB\n", ws_floats * 4 / 1e6);

    gate.speedup(base_ms, flash_ms, 2.5, "3-kernel cuBLAS");
    if (!(el.first_bad == size_t(-1) && !el.nan_count)) {
        std::printf("note           : L is wrong, so exercise 20 cannot use this.\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO)); CUDA_CHECK(cudaFree(dOref));
    CUDA_CHECK(cudaFree(dL)); CUDA_CHECK(cudaFree(dLref));
    CUDA_CHECK(cudaFree(dS)); CUDA_CHECK(cudaFree(dP)); CUDA_CHECK(cudaFree(dWs));
    return gate.finish();
}
