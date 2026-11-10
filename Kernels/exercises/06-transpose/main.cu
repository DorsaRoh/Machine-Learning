// Harness for 06-transpose. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 8192;
    size_t elems = size_t(n) * n;
    size_t bytes = elems * sizeof(float);

    bench::Gate gate("06-transpose");

    std::vector<float> h_in(elems), h_out(elems), want(elems);
    bench::fill_random(h_in, 11);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) want[size_t(j) * n + i] = h_in[size_t(i) * n + j];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    double ceiling = bench::copy_ceiling_gbs();
    double moved = 2.0 * bytes;
    double best = 0;
    bool all_correct = true;
    bench::ErrorStats worst;

    std::printf("\n version | correct |     ms |     GB/s | %% of ceiling | vs v1\n");
    std::printf("---------+---------+--------+----------+---------------+-------\n");
    double v1_gbs = 0;
    for (int v = 1; v <= 3; ++v) {
        CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
        launch_transpose(v, d_in, d_out, n);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bench::ErrorStats s = bench::compare(h_out, want, 0.0f);
        bool ok = s.first_bad == size_t(-1) && s.nan_count == 0;
        if (!ok) { all_correct = false; worst = s; }

        double ms = bench::time_ms([&] { launch_transpose(v, d_in, d_out, n); }, 5, 30);
        double g = bench::gbs(size_t(moved), ms);
        if (v == 1) v1_gbs = g;
        if (ok && g > best) best = g;
        std::printf("      v%d | %7s | %6.3f | %8.1f | %12.0f%% | %5.2fx\n", v,
                    ok ? "yes" : "NO", ms, g, 100.0 * g / ceiling,
                    v1_gbs > 0 ? g / v1_gbs : 1.0);
    }
    std::printf("\nceiling: %.1f GB/s\n", ceiling);
    std::printf("Explain BOTH gaps: v1->v2 and v2->v3. Different mechanisms.\n");

    gate.correctness(all_correct ? bench::ErrorStats{} : worst, 0.0);
    gate.bandwidth(best, 0.85);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return gate.finish();
}
