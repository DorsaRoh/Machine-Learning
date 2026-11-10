// Harness for 02-saxpy. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);
    size_t bytes = size_t(n) * sizeof(float);
    const float alpha = 2.5f;

    bench::Gate gate("02-saxpy");

    std::vector<float> h_x(n), h_y(n), h_out(n), want(n);
    bench::fill_random(h_x, 1);
    bench::fill_random(h_y, 2);
    for (int i = 0; i < n; ++i) want[i] = alpha * h_x[i] + h_y[i];

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    launch_saxpy(alpha, d_x, d_y, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    gate.correctness(bench::compare(h_out, want, 1e-5), 1e-5);

    // y is updated in place, so timed repeats change its values. That is fine
    // for timing (same instructions, same traffic) but it is why correctness is
    // checked on a single clean run above, before the benchmark loop.
    double ms = bench::time_ms([&] { launch_saxpy(alpha, d_x, d_y, n); });
    double moved = 3.0 * bytes;  // read x, read y, write y
    std::printf("n              : %d\n", n);
    std::printf("time           : %.3f ms\n", ms);
    std::printf("compute        : %.1f GFLOP/s\n", bench::gflops(2.0 * n, ms));
    gate.bandwidth(bench::gbs(size_t(moved), ms), 0.80);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return gate.finish();
}
