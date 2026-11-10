// Harness for 08-scan. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? size_t(std::atoll(argv[1])) : (size_t(1) << 24);
    size_t bytes = n * sizeof(float);
    size_t scratch_bytes = size_t(64) << 20;  // 64 MB, plenty for any design

    bench::Gate gate("08-scan");

    std::vector<float> h_in(n), h_out(n), want(n);
    // Positive inputs: the prefix sum is monotonic, so there is no
    // cancellation and the tolerance below means what it says.
    bench::fill_random(h_in, 9, 0.0f, 1.0f);
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        want[i] = float(acc);
        acc += double(h_in[i]);
    }

    float *d_in, *d_out;
    void* d_scratch;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));

    reset_scan_state(d_scratch, scratch_bytes);
    launch_scan(d_in, d_out, d_scratch, scratch_bytes, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    gate.correctness(bench::compare(h_out, want, 1e-4), 1e-4);

    double ms = bench::time_ms(
        [&] {
            reset_scan_state(d_scratch, scratch_bytes);
            launch_scan(d_in, d_out, d_scratch, scratch_bytes, n);
        },
        3, 20);

    double ceiling = bench::copy_ceiling_gbs();
    std::printf("n              : %zu\n", n);
    std::printf("time           : %.3f ms\n", ms);
    std::printf("one-pass floor : %.3f ms  (read n + write n at %.0f GB/s)\n",
                2.0 * bytes / (ceiling * 1e9) * 1e3, ceiling);
    std::printf("3-pass floor   : %.3f ms  (what a block-sums scan costs)\n",
                6.0 * bytes / (ceiling * 1e9) * 1e3);
    gate.bandwidth(bench::gbs(2 * bytes, ms), 0.60);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scratch));
    return gate.finish();
}
