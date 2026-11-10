// Harness for 04-occupancy-sweep. Do not edit.
//
// Sweeps block size x ILP x blocks-per-SM and reports bandwidth together with
// the theoretical occupancy the CUDA occupancy API predicts, so you can see
// directly whether the two track each other. (They will not, and that is the
// lesson.)

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

template <int ILP>
static double occupancy_for(int block) {
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, stream_kernel<ILP>, block, 0));
    const bench::DeviceInfo& d = bench::device_info();
    double warps = double(max_blocks) * (block / 32);
    return warps / (d.max_threads_per_sm / 32.0);
}

static double occupancy_dispatch(int block, int ilp) {
    switch (ilp) {
        case 1: return occupancy_for<1>(block);
        case 2: return occupancy_for<2>(block);
        case 4: return occupancy_for<4>(block);
        case 8: return occupancy_for<8>(block);
    }
    return 0;
}

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? size_t(std::atoll(argv[1])) : (size_t(1) << 26);
    size_t bytes = n * sizeof(float);

    bench::Gate gate("04-occupancy-sweep");
    const bench::DeviceInfo& d = bench::device_info();

    std::vector<float> h_in(n), h_out(n), want(n);
    bench::fill_random(h_in, 3);
    for (size_t i = 0; i < n; ++i) want[i] = h_in[i] * 2.0f + 1.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // --- correctness, at an awkward ILP/grid combination ------------------
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
    launch_stream(d_in, d_out, n, 128, d.sm_count * 3 + 1, 4);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    gate.correctness(bench::compare(h_out, want, 1e-6), 1e-6);

    // --- the sweep --------------------------------------------------------
    const int blocks[] = {32, 64, 128, 256, 512, 1024};
    const int ilps[] = {1, 2, 4, 8};
    const int per_sm[] = {1, 2, 4, 8, 16};
    double ceiling = bench::copy_ceiling_gbs();
    double best = 0;
    int best_b = 0, best_i = 0, best_p = 0;

    std::printf("\n block | ILP | blocks/SM |  grid |  occ%% |     GB/s | %% ceiling\n");
    std::printf("-------+-----+-----------+-------+-------+----------+----------\n");
    for (int b : blocks) {
        for (int ilp : ilps) {
            double occ = occupancy_dispatch(b, ilp);
            for (int p : per_sm) {
                int grid = d.sm_count * p;
                double ms = bench::time_ms(
                    [&] { launch_stream(d_in, d_out, n, b, grid, ilp); }, 3, 15);
                double g = bench::gbs(2 * bytes, ms);
                if (g > best) { best = g; best_b = b; best_i = ilp; best_p = p; }
                std::printf(" %5d | %3d | %9d | %5d | %5.0f | %8.1f | %8.0f%%\n", b,
                            ilp, p, grid, 100.0 * occ, g, 100.0 * g / ceiling);
            }
        }
    }

    std::printf("\nbest: block=%d ILP=%d blocks/SM=%d -> %.1f GB/s (%.0f%% of ceiling)\n",
                best_b, best_i, best_p, best, 100.0 * best / ceiling);
    std::printf("Now: is the best config also the highest-occupancy config?\n");
    gate.bandwidth(best, 0.85);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return gate.finish();
}
