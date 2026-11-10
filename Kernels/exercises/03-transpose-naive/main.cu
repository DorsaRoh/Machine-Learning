// Harness for 03-transpose-naive. Do not edit.

#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

// A pure copy with the same traffic, as a local reference point. This is the
// speed a perfect transpose would run at.
__global__ void copy_ref(const float* __restrict__ in, float* __restrict__ out,
                         int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = in[y * n + x];
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 8192;
    size_t elems = size_t(n) * n;
    size_t bytes = elems * sizeof(float);

    bench::Gate gate("03-transpose-naive");

    std::vector<float> h_in(elems), h_out(elems), want(elems);
    bench::fill_random(h_in, 11);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) want[size_t(j) * n + i] = h_in[size_t(i) * n + j];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));

    launch_transpose(d_in, d_out, n);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    gate.correctness(bench::compare(h_out, want, 0.0f), 0.0);

    double moved = 2.0 * bytes;
    double ms = bench::time_ms([&] { launch_transpose(d_in, d_out, n); }, 5, 30);

    dim3 cb(32, 8), cg((n + 31) / 32, (n + 7) / 8);
    double copy_ms =
        bench::time_ms([&] { copy_ref<<<cg, cb>>>(d_in, d_out, n); }, 5, 30);

    double t_gbs = bench::gbs(size_t(moved), ms);
    double c_gbs = bench::gbs(size_t(moved), copy_ms);

    std::printf("matrix         : %d x %d fp32 (%.0f MB)\n", n, n, bytes / 1e6);
    std::printf("transpose      : %.3f ms -> %.1f GB/s\n", ms, t_gbs);
    std::printf("same-shape copy: %.3f ms -> %.1f GB/s\n", copy_ms, c_gbs);
    std::printf("you are paying : %.2fx over a pure copy\n", ms / copy_ms);
    std::printf("\nAccount for that factor. Sectors read per warp = ?  written = ?\n");

    gate.no_perf_gate();  // no bar here. Chapter 06 sets one.

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return gate.finish();
}
