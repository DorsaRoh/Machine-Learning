// Harness for 05-divergence. Do not edit.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

static uint32_t cpu_ref(int i, int divisor, int work) {
    uint32_t v = uint32_t(i);
    if ((i % divisor) * 2 < divisor) {
        for (int s = 0; s < work; ++s) v = v * 1664525u + 1013904223u;
    } else {
        for (int s = 0; s < work; ++s) v = v * 22695477u + 1u;
    }
    return v;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);  // 4M threads of work
    int work = (argc > 2) ? std::atoi(argv[2]) : 256;

    bench::Gate gate("05-divergence");

    uint32_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, size_t(n) * sizeof(uint32_t)));
    std::vector<uint32_t> h(n);

    // --- correctness at a maximally divergent setting ---------------------
    const int check_div = 2;
    launch_divergent(d_out, n, check_div, work);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, size_t(n) * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    bench::ErrorStats s;
    for (int i = 0; i < n; ++i) {
        uint32_t want = cpu_ref(i, check_div, work);
        if (h[i] != want) {
            if (s.first_bad == size_t(-1)) {
                s.first_bad = size_t(i);
                s.got = double(h[i]);
                s.want = double(want);
            }
            s.max_abs = 1;
            s.max_rel = 1;
        }
    }
    gate.correctness(s, 0.0);

    // --- the sweep --------------------------------------------------------
    const int divisors[] = {1, 2, 4, 8, 16, 32, 64, 128};
    double base = 0;
    std::printf("\n divisor | lanes taking A per warp |     ms | vs uniform\n");
    std::printf("---------+-------------------------+--------+-----------\n");
    for (int dv : divisors) {
        double ms = bench::time_ms(
            [&] { launch_divergent(d_out, n, dv, work); }, 5, 30, false);
        if (dv == 1) base = ms;
        int lanes_a = 0;
        for (int l = 0; l < 32; ++l)
            if ((l % dv) * 2 < dv) ++lanes_a;
        std::printf("  %6d | %23d | %6.3f | %9.2fx\n", dv, lanes_a, ms, ms / base);
    }
    std::printf("\nwork per thread: %d LCG steps\n", work);
    std::printf("Re-run with a tiny body:  make run ARGS=\"4194304 2\"\n");
    std::printf("and a huge one:           make run ARGS=\"4194304 4096\"\n");

    gate.no_perf_gate();
    CUDA_CHECK(cudaFree(d_out));
    return gate.finish();
}
