// Harness for 11-gemm-ladder. Do not edit.
//
// cuBLAS provides both the golden output and the performance baseline. It is
// column-major, so the row-major product C = A @ B is obtained by asking for
// C^T = B^T @ A^T with the operands swapped -- which costs nothing, since no
// data moves.

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

    bench::Gate gate("11-gemm-ladder");

    size_t a_elems = size_t(M) * K, b_elems = size_t(K) * N, c_elems = size_t(M) * N;
    std::vector<float> h_a(a_elems), h_b(b_elems), h_c(c_elems), want(c_elems);
    bench::fill_random(h_a, 41, -0.5f, 0.5f);
    bench::fill_random(h_b, 42, -0.5f, 0.5f);

    float *d_a, *d_b, *d_c, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_a, a_elems * 4));
    CUDA_CHECK(cudaMalloc(&d_b, b_elems * 4));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * 4));
    CUDA_CHECK(cudaMalloc(&d_ref, c_elems * 4));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_elems * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_elems * 4, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Keep the baseline honest: no TF32 behind our backs in an fp32 comparison.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    const float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&](float* out) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha, d_b, N, d_a, K, &beta, out, N));
    };

    run_cublas(d_ref);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(want.data(), d_ref, c_elems * 4, cudaMemcpyDeviceToHost));

    double flops = 2.0 * M * N * K;
    double cublas_ms = bench::time_ms([&] { run_cublas(d_ref); }, 5, 20);
    double cublas_tf = bench::tflops(flops, cublas_ms);

    std::printf("\nshape          : %d x %d x %d fp32\n", M, N, K);
    std::printf("cuBLAS         : %.3f ms -> %.2f TFLOP/s\n", cublas_ms, cublas_tf);
    std::printf("arithmetic intensity (compulsory) : %.0f FLOP/byte\n",
                flops / (4.0 * (a_elems + b_elems + c_elems)));
    std::printf("\n version | correct |  max rel |      ms |  TFLOP/s | %% cuBLAS | vs prev\n");
    std::printf("---------+---------+----------+---------+----------+----------+--------\n");

    double best_tf = 0, prev_tf = 0;
    bool v5_correct = false;
    double v5_tf = 0;
    bench::ErrorStats bad;
    bad.first_bad = 0;

    for (int v = 1; v <= IMPLEMENTED; ++v) {
        CUDA_CHECK(cudaMemset(d_c, 0, c_elems * 4));
        launch_gemm(v, d_a, d_b, d_c, M, N, K);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_elems * 4, cudaMemcpyDeviceToHost));
        bench::ErrorStats st = bench::compare(h_c, want, 1e-3);
        bool ok = st.first_bad == size_t(-1) && st.nan_count == 0;
        if (!ok) bad = st;

        double ms = bench::time_ms([&] { launch_gemm(v, d_a, d_b, d_c, M, N, K); },
                                   3, 15);
        double tf = bench::tflops(flops, ms);
        if (ok && tf > best_tf) best_tf = tf;
        if (v == 5) { v5_correct = ok; v5_tf = tf; }
        std::printf("      v%d | %7s | %8.1e | %7.3f | %8.2f | %7.0f%% | %6.2fx%s\n",
                    v, ok ? "yes" : "NO", st.max_rel, ms, tf,
                    100.0 * tf / cublas_tf, prev_tf > 0 ? tf / prev_tf : 1.0,
                    ok ? "" : "   <- WRONG");
        prev_tf = tf;
    }

    std::printf("\nThe gate is on v5. Everything before it is the climb.\n");
    gate.correctness((IMPLEMENTED >= 5 && v5_correct) ? bench::ErrorStats{} : bad,
                     1e-3);
    gate.relative(IMPLEMENTED >= 5 ? v5_tf : best_tf, cublas_tf, 0.70, "TFLOP/s",
                  "cuBLAS");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_ref));
    return gate.finish();
}
