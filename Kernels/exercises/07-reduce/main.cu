// Harness for 07-reduce. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? size_t(std::atoll(argv[1])) : (size_t(1) << 28);
    size_t bytes = n * sizeof(float);
    const int max_blocks = 65536;

    bench::Gate gate("07-reduce");

    std::vector<float> h_in(n);
    bench::fill_random(h_in, 5, 0.0f, 1.0f);  // positive: no cancellation
    // float64 reference, so the GPU is compared against the right answer and
    // not against a sequential fp32 sum (which would be LESS accurate).
    double ref = 0.0;
    for (size_t i = 0; i < n; ++i) ref += double(h_in[i]);

    float *d_in, *d_out, *d_scratch;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, size_t(max_blocks) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    double ceiling = bench::copy_ceiling_gbs();
    double best = 0, prev = 0;
    bool any_correct = false;
    bench::ErrorStats best_stats;
    best_stats.first_bad = 0;  // assume failure until a version passes
    const double tol = 1e-4;   // relative

    std::printf("\n version |          sum | rel err |     ms |     GB/s | %% ceil | vs prev\n");
    std::printf("---------+--------------+---------+--------+----------+--------+--------\n");
    for (int v = 1; v <= IMPLEMENTED; ++v) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launch_reduce(v, d_in, d_out, d_scratch, n, max_blocks);
        CUDA_CHECK_LAUNCH();
        float got = 0;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double rel = std::fabs(double(got) - ref) / std::fabs(ref);
        bool ok = rel < tol && std::isfinite(got);

        double ms = bench::time_ms(
            [&] {
                CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));
                launch_reduce(v, d_in, d_out, d_scratch, n, max_blocks);
            },
            3, 20);
        double g = bench::gbs(bytes, ms);
        if (ok) {
            any_correct = true;
            if (g > best) {
                best = g;
                best_stats = bench::ErrorStats{};
            }
        }
        std::printf("      v%d | %12.1f | %7.1e | %6.3f | %8.1f | %5.0f%% | %6.2fx%s\n",
                    v, double(got), rel, ms, g, 100.0 * g / ceiling,
                    prev > 0 ? g / prev : 1.0, ok ? "" : "   <- WRONG");
        prev = g;
    }
    std::printf("\nreference (float64): %.1f\n", ref);
    std::printf("ceiling: %.1f GB/s   perfect reduction time: %.3f ms\n", ceiling,
                bytes / (ceiling * 1e9) * 1e3);

    if (!any_correct) {
        std::printf("\nNo version produced a correct sum yet. Set IMPLEMENTED and go.\n");
    }
    gate.correctness(best_stats, tol);
    gate.bandwidth(best, 0.85);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scratch));
    return gate.finish();
}
