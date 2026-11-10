// Harness for 10-softmax. Do not edit.

#include <cmath>
#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

struct Shape {
    int rows, cols;
    const char* name;
};

int main(int argc, char** argv) {
    bench::Gate gate("10-softmax");
    double ceiling = bench::copy_ceiling_gbs();

    Shape shapes[] = {{8192, 1024, "transformer   [8192,1024]"},
                      {262144, 32, "tiny rows     [262144,32]"},
                      {64, 131072, "huge rows     [64,131072]"}};
    int nshapes = (argc > 1) ? 1 : 3;  // ARGS=1 to test only the first shape

    bool all_ok = true;
    bench::ErrorStats bad;
    double worst_frac = 1e30, worst_gbs = 0;

    std::printf("\n shape                     | correct |  max rel |     ms |     GB/s | %% ceil\n");
    std::printf("---------------------------+---------+----------+--------+----------+-------\n");

    for (int s = 0; s < nshapes; ++s) {
        int rows = shapes[s].rows, cols = shapes[s].cols;
        size_t elems = size_t(rows) * cols;
        size_t bytes = elems * sizeof(float);

        std::vector<float> h_x(elems), h_y(elems), want(elems);
        // Scale of 10 makes exp overflow without max-subtraction, so an
        // unstable implementation produces NaN rather than quietly passing.
        bench::fill_random(h_x, 21 + s, -10.0f, 10.0f);

        for (int r = 0; r < rows; ++r) {
            const float* xr = &h_x[size_t(r) * cols];
            double m = -1e300;
            for (int c = 0; c < cols; ++c) m = std::max(m, double(xr[c]));
            double sum = 0;
            for (int c = 0; c < cols; ++c) sum += std::exp(double(xr[c]) - m);
            for (int c = 0; c < cols; ++c)
                want[size_t(r) * cols + c] = float(std::exp(double(xr[c]) - m) / sum);
        }

        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_y, 0xFF, bytes));

        launch_softmax(d_x, d_y, rows, cols);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

        bench::ErrorStats st = bench::compare(h_y, want, 1e-5);
        bool ok = st.first_bad == size_t(-1) && st.nan_count == 0;
        if (!ok) { all_ok = false; bad = st; }

        double ms = bench::time_ms([&] { launch_softmax(d_x, d_y, rows, cols); }, 3, 20);
        double g = bench::gbs(2 * bytes, ms);
        double frac = g / ceiling;
        if (frac < worst_frac) { worst_frac = frac; worst_gbs = g; }

        std::printf(" %-25s | %7s | %8.1e | %6.3f | %8.1f | %5.0f%%\n",
                    shapes[s].name, ok ? "yes" : "NO", st.max_rel, ms, g,
                    100.0 * frac);

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    std::printf("\nceiling: %.1f GB/s.  The gate is on the WORST shape.\n", ceiling);
    gate.correctness(all_ok ? bench::ErrorStats{} : bad, 1e-5);
    gate.bandwidth(worst_gbs, 0.85);
    return gate.finish();
}
