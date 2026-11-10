// Harness for 10-layernorm. Do not edit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    int rows = (argc > 1) ? std::atoi(argv[1]) : 8192;
    int cols = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const float eps = 1e-5f;
    size_t elems = size_t(rows) * cols;
    size_t bytes = elems * sizeof(float);

    bench::Gate gate("10-layernorm");
    double ceiling = bench::copy_ceiling_gbs();

    std::vector<float> h_x(elems), h_y(elems), want(elems);
    std::vector<float> gamma(cols), beta(cols);
    bench::fill_random(gamma, 31);
    bench::fill_random(beta, 32);

    float *d_x, *d_y, *d_g, *d_b;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_g, size_t(cols) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, size_t(cols) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_g, gamma.data(), size_t(cols) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, beta.data(), size_t(cols) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // offset 0.0 : well conditioned.  offset 1000.0 : the cancellation trap.
    const float offsets[] = {0.0f, 1000.0f};
    const char* names[] = {"x ~ N(0,1)     ", "x ~ N(1000,1)  "};
    bool all_ok = true;
    bench::ErrorStats bad;
    double perf_gbs = 0;

    std::printf("\n input            | correct |  max rel |     ms |     GB/s | %% ceil\n");
    std::printf("------------------+---------+----------+--------+----------+-------\n");

    for (int k = 0; k < 2; ++k) {
        std::mt19937 rng(77 + k);
        std::normal_distribution<float> nd(offsets[k], 1.0f);
        for (size_t i = 0; i < elems; ++i) h_x[i] = nd(rng);

        for (int r = 0; r < rows; ++r) {
            const float* xr = &h_x[size_t(r) * cols];
            double mu = 0;
            for (int c = 0; c < cols; ++c) mu += double(xr[c]);
            mu /= cols;
            double var = 0;
            for (int c = 0; c < cols; ++c) {
                double d = double(xr[c]) - mu;
                var += d * d;
            }
            var /= cols;
            double rstd = 1.0 / std::sqrt(var + eps);
            for (int c = 0; c < cols; ++c)
                want[size_t(r) * cols + c] =
                    float((double(xr[c]) - mu) * rstd * double(gamma[c]) +
                          double(beta[c]));
        }

        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0xFF, bytes));
        launch_layernorm(d_x, d_g, d_b, d_y, rows, cols, eps);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

        bench::ErrorStats st = bench::compare(h_y, want, 1e-4);
        bool ok = st.first_bad == size_t(-1) && st.nan_count == 0;
        if (!ok) { all_ok = false; bad = st; }

        double ms = bench::time_ms(
            [&] { launch_layernorm(d_x, d_g, d_b, d_y, rows, cols, eps); }, 3, 20);
        double g = bench::gbs(2 * bytes, ms);
        if (k == 0) perf_gbs = g;
        std::printf(" %s | %7s | %8.1e | %6.3f | %8.1f | %5.0f%%\n", names[k],
                    ok ? "yes" : "NO", st.max_rel, ms, g, 100.0 * g / ceiling);
    }

    std::printf("\nshape [%d, %d], ceiling %.1f GB/s\n", rows, cols, ceiling);
    std::printf("Row 2 is the cancellation test. If it failed and row 1 passed,\n");
    std::printf("your variance formula is the problem, not your kernel.\n");

    gate.correctness(all_ok ? bench::ErrorStats{} : bad, 1e-4);
    gate.bandwidth(perf_gbs, 0.85);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));
    return gate.finish();
}
