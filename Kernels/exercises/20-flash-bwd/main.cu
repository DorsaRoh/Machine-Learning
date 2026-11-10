// Harness for 20-flash-bwd. Do not edit.
//
// Reference: the unfused five-matmul backward through cuBLAS, materializing
// [N, N] exactly as the textbook derivation does. It is both the oracle and
// the baseline you must beat.
//
// The forward is run for you, so this exercise does not depend on a working
// 19-flash-fwd.

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

__global__ void ref_softmax(const float* __restrict__ S, half* __restrict__ P,
                            float* __restrict__ L, int N, int causal) {
    int row = blockIdx.x;
    int i = row % N;
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

__global__ void ref_rowdot(const half* __restrict__ a, const half* __restrict__ b,
                           float* __restrict__ out, int d) {
    int row = blockIdx.x;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        acc += __half2float(a[size_t(row) * d + c]) * __half2float(b[size_t(row) * d + c]);
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (threadIdx.x == 0) out[row] = acc;
}

// In place: Ph holds P on entry, dS on exit. S32 holds dP.
__global__ void ref_dS(half* __restrict__ Ph, const float* __restrict__ S32,
                       const float* __restrict__ D, size_t total, int N, float scale,
                       int causal) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t row = idx / N;
    int j = int(idx % N);
    int i = int(row % N);
    float p = __half2float(Ph[idx]);
    float v = (causal && j > i) ? 0.0f : p * (S32[idx] - D[row]) * scale;
    Ph[idx] = __float2half(v);
}

struct Ref {
    cublasHandle_t h;
    float* buf32;
    half* bufh;
    float* D;
    int BH, N, d;
    float scale;
    bool causal;

    void forward(const half* Q, const half* K, const half* V, half* O, float* L) {
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &scale, K, CUDA_R_16F, d,
            (long long)N * d, Q, CUDA_R_16F, d, (long long)N * d, &zero, buf32,
            CUDA_R_32F, N, (long long)N * N, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        ref_softmax<<<BH * N, 256>>>(buf32, bufh, L, N, causal ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &one, V, CUDA_R_16F, d,
            (long long)N * d, bufh, CUDA_R_16F, N, (long long)N * N, &zero, O,
            CUDA_R_16F, d, (long long)N * d, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Assumes bufh still holds P from forward().
    void backward(const half* Q, const half* K, const half* V, const half* O,
                  const half* dO, float* dQ, float* dK, float* dV) {
        const float one = 1.0f, zero = 0.0f;
        // dV = P^T dO
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_N, CUBLAS_OP_T, d, N, N, &one, dO, CUDA_R_16F, d,
            (long long)N * d, bufh, CUDA_R_16F, N, (long long)N * N, &zero, dV,
            CUDA_R_32F, d, (long long)N * d, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // dP = dO V^T   (into buf32, destroying S)
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &one, V, CUDA_R_16F, d,
            (long long)N * d, dO, CUDA_R_16F, d, (long long)N * d, &zero, buf32,
            CUDA_R_32F, N, (long long)N * N, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        ref_rowdot<<<BH * N, 32>>>(dO, O, D, d);
        CUDA_CHECK(cudaGetLastError());
        size_t total = size_t(BH) * N * N;
        ref_dS<<<int((total + 255) / 256), 256>>>(bufh, buf32, D, total, N, scale,
                                                  causal ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
        // dQ = dS K
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &one, K, CUDA_R_16F, d,
            (long long)N * d, bufh, CUDA_R_16F, N, (long long)N * N, &zero, dQ,
            CUDA_R_32F, d, (long long)N * d, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // dK = dS^T Q
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_N, CUBLAS_OP_T, d, N, N, &one, Q, CUDA_R_16F, d,
            (long long)N * d, bufh, CUDA_R_16F, N, (long long)N * N, &zero, dK,
            CUDA_R_32F, d, (long long)N * d, BH, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
};

static void upload(const std::vector<float>& src, half* dst) {
    std::vector<half> t(src.size());
    for (size_t i = 0; i < src.size(); ++i) t[i] = __float2half(src[i]);
    CUDA_CHECK(cudaMemcpy(dst, t.data(), t.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? std::atoi(argv[1]) : 1;
    int H = (argc > 2) ? std::atoi(argv[2]) : 8;
    int N = (argc > 3) ? std::atoi(argv[3]) : 2048;
    int d = (argc > 4) ? std::atoi(argv[4]) : 64;
    bool causal = (argc > 5) ? std::atoi(argv[5]) != 0 : false;

    int BH = B * H;
    float scale = 1.0f / std::sqrt(float(d));
    bench::print_device();
    bench::Gate gate("20-flash-bwd");

    size_t qkv = size_t(BH) * N * d;
    std::vector<float> hq(qkv), hk(qkv), hv(qkv), hdo(qkv);
    bench::fill_random(hq, 11, -1.f, 1.f);
    bench::fill_random(hk, 22, -1.f, 1.f);
    bench::fill_random(hv, 33, -1.f, 1.f);
    bench::fill_random(hdo, 44, -1.f, 1.f);

    half *dQh, *dKh, *dVh, *dOh, *ddOh, *bufh;
    float *L, *buf32, *D, *dQ, *dK, *dV, *rQ, *rK, *rV, *ws;
    CUDA_CHECK(cudaMalloc(&dQh, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dKh, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dVh, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOh, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&ddOh, qkv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&L, size_t(BH) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D, size_t(BH) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf32, size_t(BH) * N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bufh, size_t(BH) * N * N * sizeof(half)));
    for (float** p : {&dQ, &dK, &dV, &rQ, &rK, &rV})
        CUDA_CHECK(cudaMalloc(p, qkv * sizeof(float)));
    size_t ws_floats = size_t(BH) * N * (d + 8);
    CUDA_CHECK(cudaMalloc(&ws, ws_floats * sizeof(float)));

    upload(hq, dQh); upload(hk, dKh); upload(hv, dVh); upload(hdo, ddOh);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    Ref ref{handle, buf32, bufh, D, BH, N, d, scale, causal};

    ref.forward(dQh, dKh, dVh, dOh, L);
    ref.backward(dQh, dKh, dVh, dOh, ddOh, rQ, rK, rV);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (float* p : {dQ, dK, dV}) CUDA_CHECK(cudaMemset(p, 0, qkv * sizeof(float)));
    launch_flash_bwd(dQh, dKh, dVh, dOh, ddOh, L, dQ, dK, dV, BH, N, d, scale,
                     causal, ws, ws_floats);
    CUDA_CHECK_LAUNCH();

    const char* names[3] = {"dQ", "dK", "dV"};
    float* mine[3] = {dQ, dK, dV};
    float* theirs[3] = {rQ, rK, rV};
    bool ok = true;
    bench::ErrorStats worst;
    std::vector<float> a(qkv), b(qkv);
    std::printf("\n");
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemcpy(a.data(), mine[i], qkv * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), theirs[i], qkv * 4, cudaMemcpyDeviceToHost));
        bench::ErrorStats s = bench::compare(a, b, 3e-3);
        bool good = s.first_bad == size_t(-1) && s.nan_count == 0;
        std::printf("%s : max_abs %.3g  max_rel %.3g  %s\n", names[i], s.max_abs,
                    s.max_rel, good ? "ok" : "WRONG");
        if (!good) { ok = false; worst = s; }
    }
    gate.correctness(ok ? bench::ErrorStats{} : worst, 3e-3);

    double ref_ms = bench::time_ms(
        [&] {
            ref.forward(dQh, dKh, dVh, dOh, L);
            ref.backward(dQh, dKh, dVh, dOh, ddOh, rQ, rK, rV);
        },
        3, 15, false);
    double fwd_ms = bench::time_ms([&] { ref.forward(dQh, dKh, dVh, dOh, L); }, 3, 15, false);
    double bwd_ref_ms = ref_ms - fwd_ms;
    double mine_ms = bench::time_ms(
        [&] {
            launch_flash_bwd(dQh, dKh, dVh, dOh, ddOh, L, dQ, dK, dV, BH, N, d,
                             scale, causal, ws, ws_floats);
        },
        3, 15, false);

    double flops = 10.0 * double(BH) * N * N * d * (causal ? 0.5 : 1.0);
    std::printf("\nshape          : B=%d H=%d N=%d d=%d causal=%d\n", B, H, N, d,
                int(causal));
    std::printf("backward FLOPs : %.2f GFLOP (5 matmuls)\n", flops / 1e9);
    std::printf("unfused ref    : %.3f ms  (%.1f TFLOP/s)\n", bwd_ref_ms,
                bench::tflops(flops, bwd_ref_ms));
    std::printf("yours          : %.3f ms  (%.1f TFLOP/s)\n", mine_ms,
                bench::tflops(flops, mine_ms));
    std::printf("[N,N] the ref materializes : %.0f MB\n",
                double(BH) * N * N * 6 / 1e6);

    gate.speedup(bwd_ref_ms, mine_ms, 2.0, "unfused cuBLAS backward");

    CUBLAS_CHECK(cublasDestroy(handle));
    return gate.finish();
}
