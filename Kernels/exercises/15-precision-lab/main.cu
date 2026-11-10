// Harness for 15-precision-lab. Do not edit.
//
// Everything here is ground truth in float64 on the CPU. The GPU numbers are
// compared against it, and the comparison is the exercise.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

static double rel(double got, double want) {
    return std::fabs(got - want) / std::max(1e-300, std::fabs(want));
}

int main() {
    bench::Gate gate("15-precision-lab");
    std::printf("\nNo performance gate here. The numbers are the exercise.\n");

    // ---------------------------------------------------------------- E1
    {
        size_t n = size_t(1) << 24;
        std::vector<float> h(n);
        // Deliberately awkward: values spanning many orders of magnitude, so
        // a naive sequential sum loses the small ones entirely.
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> big(0.9f, 1.1f);
        std::uniform_real_distribution<float> small(1e-8f, 1e-7f);
        for (size_t i = 0; i < n; ++i) h[i] = (i % 1000 == 0) ? big(rng) : small(rng);

        double truth = 0;  // float64, in order; good enough as ground truth here
        for (size_t i = 0; i < n; ++i) truth += double(h[i]);

        float* d;
        double* d_out;
        CUDA_CHECK(cudaMalloc(&d, n * 4));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));

        const char* names[] = {"sequential fp32", "tree fp32      ",
                               "Kahan fp32     ", "tree fp64      "};
        std::printf("\nE1  accumulation order   (n = %zu, truth = %.12e)\n", n, truth);
        std::printf("    method              |        result | rel error\n");
        std::printf("    --------------------+---------------+----------\n");
        for (int m = 0; m < 4; ++m) {
            double got = 0;
            CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
            launch_sum(d, d_out, n, m);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(double), cudaMemcpyDeviceToHost));
            std::printf("    %s     | %13.6e | %8.2e\n", names[m], got, rel(got, truth));
        }
        CUDA_CHECK(cudaFree(d));
        CUDA_CHECK(cudaFree(d_out));
    }

    // ---------------------------------------------------------------- E2
    {
        int rows = 1024, cols = 4096;
        size_t e = size_t(rows) * cols;
        std::vector<float> h(e), got(rows);
        std::vector<double> truth(rows);
        const float offsets[] = {0.0f, 100.0f, 10000.0f};

        float *d_x, *d_v;
        CUDA_CHECK(cudaMalloc(&d_x, e * 4));
        CUDA_CHECK(cudaMalloc(&d_v, size_t(rows) * 4));

        std::printf("\nE2  cancellation         (variance of x ~ N(mu, 1), %d cols)\n", cols);
        std::printf("    mu      | identity fp32 | Welford fp32 | identity fp64\n");
        std::printf("    --------+---------------+--------------+--------------\n");
        for (float mu : offsets) {
            std::mt19937 rng(11);
            std::normal_distribution<float> nd(mu, 1.0f);
            for (size_t i = 0; i < e; ++i) h[i] = nd(rng);
            for (int r = 0; r < rows; ++r) {
                const float* xr = &h[size_t(r) * cols];
                double m = 0;
                for (int c = 0; c < cols; ++c) m += double(xr[c]);
                m /= cols;
                double v = 0;
                for (int c = 0; c < cols; ++c) v += (double(xr[c]) - m) * (double(xr[c]) - m);
                truth[r] = v / cols;
            }
            CUDA_CHECK(cudaMemcpy(d_x, h.data(), e * 4, cudaMemcpyHostToDevice));
            double errs[3];
            for (int m = 0; m < 3; ++m) {
                launch_variance(d_x, d_v, rows, cols, m);
                CUDA_CHECK_LAUNCH();
                CUDA_CHECK(cudaMemcpy(got.data(), d_v, size_t(rows) * 4,
                                      cudaMemcpyDeviceToHost));
                double worst = 0;
                for (int r = 0; r < rows; ++r)
                    worst = std::max(worst, rel(double(got[r]), truth[r]));
                errs[m] = worst;
            }
            std::printf("    %7.0f | %13.2e | %12.2e | %12.2e\n", mu, errs[0], errs[1],
                        errs[2]);
        }
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_v));
    }

    // ---------------------------------------------------------------- E3
    {
        std::printf("\nE3  accumulator width    (fp16 inputs, dot product)\n");
        std::printf("         n | fp32 acc err | fp16 acc err\n");
        std::printf("    -------+--------------+-------------\n");
        for (int n : {128, 1024, 8192, 65536, 1048576}) {
            std::vector<float> fa(n), fb(n);
            bench::fill_random(fa, 3, -1.0f, 1.0f);
            bench::fill_random(fb, 4, -1.0f, 1.0f);
            std::vector<half> ha(n), hb(n);
            double truth = 0;
            for (int i = 0; i < n; ++i) {
                ha[i] = __float2half(fa[i]);
                hb[i] = __float2half(fb[i]);
                truth += double(__half2float(ha[i])) * double(__half2float(hb[i]));
            }
            half *da, *db;
            float* dout;
            CUDA_CHECK(cudaMalloc(&da, size_t(n) * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&db, size_t(n) * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
            CUDA_CHECK(cudaMemcpy(da, ha.data(), size_t(n) * sizeof(half),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(db, hb.data(), size_t(n) * sizeof(half),
                                  cudaMemcpyHostToDevice));
            double errs[2];
            for (int m = 0; m < 2; ++m) {
                float got = 0;
                CUDA_CHECK(cudaMemset(dout, 0, sizeof(float)));
                launch_dot(da, db, dout, n, m);
                CUDA_CHECK_LAUNCH();
                CUDA_CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
                errs[m] = rel(double(got), truth);
            }
            std::printf("    %6d | %12.2e | %12.2e\n", n, errs[0], errs[1]);
            CUDA_CHECK(cudaFree(da));
            CUDA_CHECK(cudaFree(db));
            CUDA_CHECK(cudaFree(dout));
        }
    }

    // ---------------------------------------------------------------- E4
    {
        const float probes[] = {1e-5f, 1e-3f, 1.0f, 100.0f, 255.0f, 300.0f, 1e4f, 1e6f};
        int n = sizeof(probes) / sizeof(float);
        std::vector<float> h(probes, probes + n), got(n);
        float *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dx, size_t(n) * 4));
        CUDA_CHECK(cudaMalloc(&dy, size_t(n) * 4));
        CUDA_CHECK(cudaMemcpy(dx, h.data(), size_t(n) * 4, cudaMemcpyHostToDevice));

        std::printf("\nE4  format range         (y = x*x, computed and stored in format)\n");
        std::printf("            x |     fp32 |     fp16 |     bf16\n");
        std::printf("    ----------+----------+----------+---------\n");
        std::vector<std::vector<float>> res(3, std::vector<float>(n));
        for (int m = 0; m < 3; ++m) {
            launch_square(dx, dy, n, m);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(res[m].data(), dy, size_t(n) * 4,
                                  cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < n; ++i)
            std::printf("    %9.1e | %8.2e | %8.2e | %8.2e\n", probes[i], res[0][i],
                        res[1][i], res[2][i]);
        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dy));
    }

    std::printf("\nNow explain every row. The explanations are the exercise.\n");
    gate.correctness(bench::ErrorStats{}, 0.0);
    gate.no_perf_gate();
    return gate.finish();
}
