// Harness for 16-diagnose. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 8192;  // square side for k2
    int rows = n, cols = n;
    size_t elems = size_t(n) * n;
    size_t bytes = elems * sizeof(float);

    bench::Gate gate("16-diagnose");
    double ceiling = bench::copy_ceiling_gbs();

    std::vector<float> h_in(elems), h_a(elems), h_b(elems);
    bench::fill_random(h_in, 71);

    float *d_in, *d_a, *d_b, *d_rows;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_rows, size_t(rows) * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // bars: fraction of copy ceiling each FIXED kernel must reach
    const double bars[3] = {0.80, 0.80, 0.80};
    const char* names[3] = {"1 row sums  ", "2 transpose ", "3 affine x8 "};
    double traffic[3] = {double(bytes) + double(rows) * 4, 2.0 * double(bytes),
                         2.0 * double(bytes)};
    bool ok[3] = {false, false, false};
    double frac[3] = {0, 0, 0};
    bench::ErrorStats bad;
    bad.first_bad = 0;
    bool all_ok = true;

    std::printf("\n kernel       |   slow ms |  slow %%ceil |   fast ms |  fast %%ceil | speedup | correct\n");
    std::printf("--------------+-----------+-------------+-----------+-------------+---------+--------\n");

    for (int k = 0; k < 3; ++k) {
        auto run = [&](bool fixed, float* out) {
            if (k == 0) launch_1(fixed, d_in, d_rows + (fixed ? rows : 0), rows, cols);
            else if (k == 1) launch_2(fixed, d_in, out, n);
            else launch_3(fixed, d_in, out, n * n);
        };

        CUDA_CHECK(cudaMemset(d_a, 0, bytes));
        CUDA_CHECK(cudaMemset(d_b, 0, bytes));
        CUDA_CHECK(cudaMemset(d_rows, 0, size_t(rows) * 2 * sizeof(float)));
        run(false, d_a);
        CUDA_CHECK_LAUNCH();
        run(true, d_b);
        CUDA_CHECK_LAUNCH();

        // correctness: fixed must match slow, elementwise
        size_t count = (k == 0) ? size_t(rows) : elems;
        std::vector<float> ga(count), gb(count);
        const float* src_a = (k == 0) ? d_rows : d_a;
        const float* src_b = (k == 0) ? d_rows + rows : d_b;
        CUDA_CHECK(cudaMemcpy(ga.data(), src_a, count * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gb.data(), src_b, count * 4, cudaMemcpyDeviceToHost));
        bench::ErrorStats st = bench::compare(gb, ga, 1e-4);
        ok[k] = st.first_bad == size_t(-1) && st.nan_count == 0;
        if (!ok[k]) { all_ok = false; bad = st; }

        double slow_ms = bench::time_ms([&] { run(false, d_a); }, 3, 15);
        double fast_ms = bench::time_ms([&] { run(true, d_b); }, 3, 15);
        double slow_g = bench::gbs(size_t(traffic[k]), slow_ms);
        double fast_g = bench::gbs(size_t(traffic[k]), fast_ms);
        frac[k] = fast_g / ceiling;

        std::printf(" %s | %9.3f | %10.0f%% | %9.3f | %10.0f%% | %6.2fx | %7s\n",
                    names[k], slow_ms, 100.0 * slow_g / ceiling, fast_ms,
                    100.0 * frac[k], slow_ms / fast_ms, ok[k] ? "yes" : "NO");
    }

    std::printf("\nceiling %.1f GB/s. Each fixed kernel must reach 80%% of it.\n", ceiling);
    double worst = 1e30;
    for (int k = 0; k < 3; ++k) worst = std::min(worst, frac[k] / bars[k]);
    std::printf("Write your three diagnoses down BEFORE you look at the fixes.\n");

    gate.correctness(all_ok ? bench::ErrorStats{} : bad, 1e-4);
    gate.bandwidth(worst * bars[0] * ceiling, bars[0]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_rows));
    return gate.finish();
}
