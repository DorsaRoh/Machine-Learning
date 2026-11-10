// Harness for 13-tensor-core-gemm. Do not edit.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

#define CUBLAS_CHECK(x)                                                      \
    do {                                                                     \
        cublasStatus_t s = (x);                                              \
        if (s != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s),       \
                         __FILE__, __LINE__);                                \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int N = (argc > 2) ? std::atoi(argv[2]) : M;
    int K = (argc > 3) ? std::atoi(argv[3]) : M;

    bench::Gate gate("13-tensor-core-gemm");

    size_t ae = size_t(M) * K, be = size_t(K) * N, ce = size_t(M) * N;
    std::vector<float> f_a(ae), f_b(be), h_c(ce), want(ce);
    bench::fill_random(f_a, 51, -0.5f, 0.5f);
    bench::fill_random(f_b, 52, -0.5f, 0.5f);
    std::vector<half> h_a(ae), h_b(be);
    for (size_t i = 0; i < ae; ++i) h_a[i] = __float2half(f_a[i]);
    for (size_t i = 0; i < be; ++i) h_b[i] = __float2half(f_b[i]);

    half *d_a, *d_b;
    float *d_c, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_a, ae * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, be * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, ce * 4));
    CUDA_CHECK(cudaMalloc(&d_ref, ce * 4));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), ae * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), be * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&](float* out) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K,
                                  &beta, out, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    run_cublas(d_ref);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(want.data(), d_ref, ce * 4, cudaMemcpyDeviceToHost));

    double flops = 2.0 * M * N * K;
    double cublas_ms = bench::time_ms([&] { run_cublas(d_ref); }, 5, 20);
    double cublas_tf = bench::tflops(flops, cublas_ms);

    std::printf("\nshape          : %d x %d x %d, fp16 in / fp32 accumulate\n", M, N, K);
    std::printf("cuBLAS (TC)    : %.3f ms -> %.2f TFLOP/s\n", cublas_ms, cublas_tf);
    std::printf("fp16 knee      : %.0f FLOP/byte; this problem has %.0f\n",
                cublas_tf * 1e12 / (bench::copy_ceiling_gbs() * 1e9),
                flops / (2.0 * (ae + be) + 4.0 * ce));
    std::printf("\n version | correct |  max rel |      ms |  TFLOP/s | %% cuBLAS\n");
    std::printf("---------+---------+----------+---------+----------+---------\n");

    double best = 0;
    bool any_ok = false;
    bench::ErrorStats bad;
    bad.first_bad = 0;
    const double tol = 2e-2;  // fp16 inputs: this is the right order of tolerance

    for (int v = 1; v <= IMPLEMENTED; ++v) {
        CUDA_CHECK(cudaMemset(d_c, 0, ce * 4));
        launch_tc_gemm(v, d_a, d_b, d_c, M, N, K);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, ce * 4, cudaMemcpyDeviceToHost));
        bench::ErrorStats st = bench::compare(h_c, want, tol);
        bool ok = st.first_bad == size_t(-1) && st.nan_count == 0;
        if (!ok) bad = st;
        double ms = bench::time_ms([&] { launch_tc_gemm(v, d_a, d_b, d_c, M, N, K); },
                                   3, 15);
        double tf = bench::tflops(flops, ms);
        if (ok) { any_ok = true; if (tf > best) best = tf; }
        std::printf("      v%d | %7s | %8.1e | %7.3f | %8.2f | %7.0f%%\n", v,
                    ok ? "yes" : "NO", st.max_rel, ms, tf, 100.0 * tf / cublas_tf);
    }

    std::printf("\nIf TFLOP/s is near your fp32 ladder's number, check the SASS\n");
    std::printf("for HMMA. No HMMA means no tensor cores ran.\n");

    gate.correctness(any_ok ? bench::ErrorStats{} : bad, tol);
    gate.relative(best, cublas_tf, 0.50, "TFLOP/s", "cuBLAS fp16");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_ref));
    return gate.finish();
}
