// Harness for 02-vector-add. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);  // 64M floats
    size_t bytes = size_t(n) * sizeof(float);

    bench::Gate gate("02-vector-add");

    std::vector<float> h_a(n), h_b(n), h_c(n), want(n);
    bench::fill_random(h_a, 1);
    bench::fill_random(h_b, 2);
    for (int i = 0; i < n; ++i) want[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0xFF, bytes));

    launch_vector_add(d_a, d_b, d_c, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    gate.correctness(bench::compare(h_c, want, 0.0f), 0.0);

    double ms = bench::time_ms([&] { launch_vector_add(d_a, d_b, d_c, n); });
    // 3 streams of traffic: read a, read b, write c.
    double moved = 3.0 * bytes;
    std::printf("n              : %d (%.0f MB per array)\n", n, bytes / 1e6);
    std::printf("time           : %.3f ms\n", ms);
    std::printf("compute        : %.1f GFLOP/s  (%.2f%% of fp32 peak you looked up)\n",
                bench::gflops(double(n), ms), 0.0);
    gate.bandwidth(bench::gbs(size_t(moved), ms), 0.80);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return gate.finish();
}
