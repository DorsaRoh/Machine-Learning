// Harness for 22-moe-dispatch. Do not edit.
//
// Reference: the dense-masked version. Every token through every expert,
// weighted by a [T, E] router matrix. Trivially correct, E/KTOP times too much
// work, and completely regular -- which is why it is not as slow as it sounds
// and why beating it by the full E/KTOP is not free.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <random>
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

__global__ void accumulate_masked(const float* __restrict__ out,
                                  const float* __restrict__ wmat,
                                  float* __restrict__ y, int T, int H, int E, int e) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * H) return;
    int t = int(i / H);
    float w = wmat[size_t(t) * E + e];
    if (w != 0.0f) y[i] += w * out[i];
}

int main(int argc, char** argv) {
    int T = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int H = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int E = (argc > 3) ? std::atoi(argv[3]) : 8;
    int ktop = (argc > 4) ? std::atoi(argv[4]) : 2;
    double skew = (argc > 5) ? std::atof(argv[5]) : 0.0;  // 0 = uniform router
    if (skew > 0.9) skew = 0.9;
    if (ktop > E) ktop = E;

    bench::print_device();
    bench::Gate gate("22-moe-dispatch");

    // ---- router -----------------------------------------------------------
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<int> ids(size_t(T) * ktop);
    std::vector<float> pr(size_t(T) * ktop), wmat(size_t(T) * E, 0.f);
    std::vector<int> truth(E, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<int> picked;
        for (int s = 0; s < ktop; ++s) {
            int e;
            do {
                // skew > 0 biases towards expert 0, the way a collapsed router does
                e = (u(rng) < skew) ? 0 : int(u(rng) * E) % E;
            } while (std::find(picked.begin(), picked.end(), e) != picked.end());
            picked.push_back(e);
            float p = 0.2f + 0.6f * u(rng);
            ids[size_t(t) * ktop + s] = e;
            pr[size_t(t) * ktop + s] = p;
            wmat[size_t(t) * E + e] = p;
            truth[e]++;
        }
    }
    std::printf("\nrouter load (tokens per expert):");
    for (int e = 0; e < E; ++e) std::printf(" %d", truth[e]);
    std::printf("\n  max/mean = %.2f\n",
                double(*std::max_element(truth.begin(), truth.end())) /
                    (double(T) * ktop / E));

    // ---- data -------------------------------------------------------------
    size_t xn = size_t(T) * H, wn = size_t(E) * H * H;
    std::vector<float> xf(xn), wf(wn);
    bench::fill_random(xf, 1, -1.f, 1.f);
    bench::fill_random(wf, 2, -0.05f, 0.05f);
    std::vector<half> xh(xn), wh(wn);
    for (size_t i = 0; i < xn; ++i) xh[i] = __float2half(xf[i]);
    for (size_t i = 0; i < wn; ++i) wh[i] = __float2half(wf[i]);

    half *d_x, *d_w;
    float *d_y, *d_ref, *d_out, *d_pr, *d_wmat;
    int *d_ids;
    void* d_ws;
    CUDA_CHECK(cudaMalloc(&d_x, xn * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, wn * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, xn * 4));
    CUDA_CHECK(cudaMalloc(&d_ref, xn * 4));
    CUDA_CHECK(cudaMalloc(&d_out, xn * 4));
    CUDA_CHECK(cudaMalloc(&d_ids, ids.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_pr, pr.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_wmat, wmat.size() * 4));
    size_t ws_bytes = size_t(T) * ktop * H * sizeof(half) +
                      size_t(T) * ktop * H * sizeof(float) +
                      size_t(T) * ktop * 8 * sizeof(int) + (1 << 16);
    CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, xh.data(), xn * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, wh.data(), wn * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pr, pr.data(), pr.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wmat, wmat.data(), wmat.size() * 4, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- dense-masked reference -------------------------------------------
    const float one = 1.0f, zero = 0.0f;
    auto dense = [&](float* dst) {
        CUDA_CHECK(cudaMemset(dst, 0, xn * 4));
        for (int e = 0; e < E; ++e) {
            // out[T,H] = x[T,H] @ W_e[H,H], row-major via column-major swap
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, T, H,
                                      &one, d_w + size_t(e) * H * H, CUDA_R_16F, H,
                                      d_x, CUDA_R_16F, H, &zero, d_out, CUDA_R_32F,
                                      H, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            accumulate_masked<<<int((xn + 255) / 256), 256>>>(d_out, d_wmat, dst, T,
                                                              H, E, e);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    dense(d_ref);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_y, 0, xn * 4));
    launch_moe(d_x, d_w, d_ids, d_pr, d_y, T, H, E, ktop, d_ws, ws_bytes, handle);
    CUDA_CHECK_LAUNCH();

    std::vector<float> got(xn), want(xn);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, xn * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(want.data(), d_ref, xn * 4, cudaMemcpyDeviceToHost));
    gate.correctness(bench::compare(got, want, 2e-2), 2e-2);

    double dense_ms = bench::time_ms([&] { dense(d_ref); }, 3, 15, false);
    double mine_ms = bench::time_ms(
        [&] {
            launch_moe(d_x, d_w, d_ids, d_pr, d_y, T, H, E, ktop, d_ws, ws_bytes,
                       handle);
        },
        3, 15, false);

    double dense_flops = 2.0 * E * double(T) * H * H;
    double sparse_flops = 2.0 * double(T) * ktop * H * H;
    double ideal = double(ktop) / E;
    std::printf("\nshape          : T=%d H=%d E=%d ktop=%d skew=%.2f\n", T, H, E,
                ktop, skew);
    std::printf("dense FLOPs    : %.1f GFLOP\n", dense_flops / 1e9);
    std::printf("needed FLOPs   : %.1f GFLOP  (%.0f%% of dense)\n",
                sparse_flops / 1e9, 100.0 * ideal);
    std::printf("dense-masked   : %.3f ms  (%.1f TFLOP/s)\n", dense_ms,
                bench::tflops(dense_flops, dense_ms));
    std::printf("yours          : %.3f ms  (%.1f TFLOP/s on useful work)\n", mine_ms,
                bench::tflops(sparse_flops, mine_ms));
    std::printf("perfect dispatch would be : %.3f ms\n", dense_ms * ideal);

    gate.speedup(dense_ms, mine_ms, 0.70 * double(E) / ktop, "dense-masked");

    CUBLAS_CHECK(cublasDestroy(handle));
    return gate.finish();
}
