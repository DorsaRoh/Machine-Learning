// Harness for 03-stride-study. Do not edit.
//
// Sweeps the stride while holding the SOURCE BUFFER size fixed at 256 MB. The
// number of elements read therefore falls as 1/stride, which is what you want:
// the useful work shrinks, but the hardware traffic does not, and the gap
// between those two is exactly the thing being measured.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    const size_t total = size_t(1) << 26;  // 64M floats = 256 MB source
    size_t src_bytes = total * sizeof(float);

    bench::Gate gate("03-stride-study");

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, src_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, src_bytes));

    std::vector<float> h_in(total);
    bench::fill_random(h_in, 7);
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), src_bytes, cudaMemcpyHostToDevice));

    // --- correctness at a stride that exercises the bounds logic ----------
    const int check_stride = 3;
    int check_n = int(total / check_stride);
    CUDA_CHECK(cudaMemset(d_out, 0xFF, src_bytes));
    launch_strided_read(d_in, d_out, check_n, check_stride);
    CUDA_CHECK_LAUNCH();

    std::vector<float> got(check_n), want(check_n);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, size_t(check_n) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < check_n; ++i) want[i] = h_in[size_t(i) * check_stride];
    gate.correctness(bench::compare(got, want, 0.0f), 0.0);

    // --- the sweep --------------------------------------------------------
    const int strides[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 1024};
    double ceiling = bench::copy_ceiling_gbs();
    double stride1_gbs = 0;

    std::printf("\n stride |   elements |  useful GB/s | %% of ceiling | vs stride 1\n");
    std::printf("--------+------------+--------------+---------------+------------\n");
    for (int s : strides) {
        int n = int(total / s);
        double ms = bench::time_ms(
            [&] { launch_strided_read(d_in, d_out, n, s); }, 5, 30);
        double useful = 2.0 * double(n) * sizeof(float);  // read + write
        double g = bench::gbs(size_t(useful), ms);
        if (s == 1) stride1_gbs = g;
        std::printf("  %5d | %10d | %12.1f | %12.0f%% | %10.2fx\n", s, n, g,
                    100.0 * g / ceiling, g / stride1_gbs);
    }
    std::printf("\nmeasured copy ceiling: %.1f GB/s\n", ceiling);
    std::printf("Now explain: where does the curve flatten, and why THERE?\n");

    gate.bandwidth(stride1_gbs, 0.75);  // the gate is on stride 1 only

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return gate.finish();
}
