// Harness for 09-histogram. Do not edit.

#include <cstdio>
#include <random>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

// Generate bin indices with a tunable fraction `p` concentrated in bin 0.
static void make_data(std::vector<unsigned>& v, int nbins, double p,
                      unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::uniform_int_distribution<int> b(0, nbins - 1);
    for (auto& x : v) x = (u(rng) < p) ? 0u : unsigned(b(rng));
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);
    int nbins = (argc > 2) ? std::atoi(argv[2]) : 256;
    size_t bytes = size_t(n) * sizeof(unsigned);

    bench::Gate gate("09-histogram");

    unsigned* d_in;
    unsigned* d_hist;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_hist, size_t(nbins) * sizeof(unsigned)));

    std::vector<unsigned> h_in(n), h_hist(nbins);
    double ceiling = bench::copy_ceiling_gbs();

    const double skews[] = {0.0, 0.5, 0.9};
    const char* names[] = {"uniform", "p=0.5 in bin 0", "p=0.9 in bin 0"};
    bool all_ok = true;
    bench::ErrorStats bad;
    double uniform_gbs = 0, worst_gbs = 1e30;

    std::printf("\n distribution    | correct |     ms |     GB/s | %% of ceiling\n");
    std::printf("-----------------+---------+--------+----------+--------------\n");
    for (int k = 0; k < 3; ++k) {
        make_data(h_in, nbins, skews[k], 100 + k);
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

        std::vector<unsigned> want(nbins, 0u);
        for (int i = 0; i < n; ++i) ++want[h_in[i]];

        CUDA_CHECK(cudaMemset(d_hist, 0, size_t(nbins) * sizeof(unsigned)));
        launch_histogram(d_in, d_hist, n, nbins);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist,
                              size_t(nbins) * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int b = 0; b < nbins; ++b)
            if (h_hist[b] != want[b]) {
                if (ok) {
                    bad.first_bad = size_t(b);
                    bad.got = double(h_hist[b]);
                    bad.want = double(want[b]);
                    bad.max_abs = std::fabs(double(h_hist[b]) - double(want[b]));
                    bad.max_rel = 1.0;
                }
                ok = false;
            }
        if (!ok) all_ok = false;

        double ms = bench::time_ms(
            [&] {
                CUDA_CHECK(cudaMemsetAsync(d_hist, 0, size_t(nbins) * sizeof(unsigned)));
                launch_histogram(d_in, d_hist, n, nbins);
            },
            3, 20);
        double g = bench::gbs(bytes, ms);
        if (k == 0) uniform_gbs = g;
        if (g < worst_gbs) worst_gbs = g;
        std::printf(" %-15s | %7s | %6.3f | %8.1f | %11.0f%%\n", names[k],
                    ok ? "yes" : "NO", ms, g, 100.0 * g / ceiling);
    }

    std::printf("\nnbins = %d, n = %d\n", nbins, n);
    std::printf("collapse factor (uniform / worst): %.2fx\n", uniform_gbs / worst_gbs);
    std::printf("A kernel that is skew-proof has a collapse factor near 1.\n");

    gate.correctness(all_ok ? bench::ErrorStats{} : bad, 0.0);
    // The gate is on the WORST distribution. Fast on easy data is not the job.
    gate.bandwidth(worst_gbs, 0.50);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_hist));
    return gate.finish();
}
