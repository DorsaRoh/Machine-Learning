// Harness for 14-pipelined-gemm. Do not edit.

#include <cublas_v2.h>

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

    bench::Gate gate("14-pipelined-gemm");
    const bench::DeviceInfo& d = bench::device_info();
    if (d.major < 8) {
        std::printf("\nNote: sm_%d%d has no cp.async. Build the register-staged\n",
                    d.major, d.minor);
        std::printf("double buffer from Chapter 12 instead. Same gate.\n");
    }

    size_t ae = size_t(M) * K, be = size_t(K) * N, ce = size_t(M) * N;
    std::vector<float> h_a(ae), h_b(be), h_c(ce), want(ce);
    bench::fill_random(h_a, 61, -0.5f, 0.5f);
    bench::fill_random(h_b, 62, -0.5f, 0.5f);

    float *d_a, *d_b, *d_c, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_a, ae * 4));
    CUDA_CHECK(cudaMalloc(&d_b, be * 4));
    CUDA_CHECK(cudaMalloc(&d_c, ce * 4));
    CUDA_CHECK(cudaMalloc(&d_ref, ce * 4));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), ae * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), be * 4, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             d_b, N, d_a, K, &beta, d_ref, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(want.data(), d_ref, ce * 4, cudaMemcpyDeviceToHost));
    double cublas_ms = bench::time_ms(
        [&] {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha, d_b, N, d_a, K, &beta, d_ref, N));
        },
        5, 20);

    double flops = 2.0 * M * N * K;
    double times[2] = {0, 0};
    bool ok[2] = {false, false};
    bench::ErrorStats bad;
    bad.first_bad = 0;
    const char* names[2] = {"baseline ", "pipelined"};

    std::printf("\n version   | correct |  max rel |      ms |  TFLOP/s | %% cuBLAS\n");
    std::printf("-----------+---------+----------+---------+----------+---------\n");
    for (int v = 0; v < 2; ++v) {
        CUDA_CHECK(cudaMemset(d_c, 0, ce * 4));
        launch_gemm(v, d_a, d_b, d_c, M, N, K);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, ce * 4, cudaMemcpyDeviceToHost));
        bench::ErrorStats st = bench::compare(h_c, want, 1e-3);
        ok[v] = st.first_bad == size_t(-1) && st.nan_count == 0;
        if (!ok[v]) bad = st;
        times[v] = bench::time_ms(
            [&] { launch_gemm(v, d_a, d_b, d_c, M, N, K); }, 3, 15);
        std::printf(" %s | %7s | %8.1e | %7.3f | %8.2f | %7.0f%%\n", names[v],
                    ok[v] ? "yes" : "NO", st.max_rel, times[v],
                    bench::tflops(flops, times[v]),
                    100.0 * cublas_ms / times[v]);
    }
    std::printf("\ncuBLAS: %.3f ms (%.2f TFLOP/s), STAGES = %d\n", cublas_ms,
                bench::tflops(flops, cublas_ms), STAGES);
    std::printf("Sweep STAGES and plot. The optimum is latency/compute, capped\n");
    std::printf("by what shared memory lets you keep resident.\n");

    gate.correctness((ok[0] && ok[1]) ? bench::ErrorStats{} : bad, 1e-3);
    gate.speedup(times[0], times[1], 1.15, "your baseline");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_ref));
    return gate.finish();
}
