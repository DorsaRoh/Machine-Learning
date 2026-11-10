// Harness for 21-int4-gemv. Do not edit.

#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

// Reads the ORIGINAL packing, always. fp32 accumulate.
__global__ void ref_gemv(const uint32_t* __restrict__ qw, const half* __restrict__ s,
                         const half* __restrict__ z, const half* __restrict__ x,
                         float* __restrict__ y, int M, int K) {
    int m = blockIdx.x;
    size_t wrow = size_t(m) * (K / 8);
    size_t grow = size_t(m) * (K / GROUP);
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        uint32_t w = qw[wrow + (k >> 3)];
        int q = int((w >> (4 * (k & 7))) & 0xFu);
        int g = k / GROUP;
        float sc = __half2float(s[grow + g]);
        float ze = __half2float(z[grow + g]);
        acc += (float(q) - ze) * sc * __half2float(x[k]);
    }
    __shared__ float red[32];
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v = (int(threadIdx.x) < nw) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) y[m] = v;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 16384;
    int K = (argc > 2) ? std::atoi(argv[2]) : 8192;
    if (K % GROUP || K % 8) {
        std::printf("K must be a multiple of %d\n", GROUP);
        return 1;
    }

    bench::print_device();
    bench::Gate gate("21-int4-gemv");

    size_t words = size_t(M) * (K / 8);
    size_t ngroups = size_t(M) * (K / GROUP);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> qd(0, 15);
    std::uniform_real_distribution<float> sd(0.004f, 0.02f);
    std::uniform_int_distribution<int> zd(6, 9);

    std::vector<uint32_t> hw(words, 0u);
    for (size_t i = 0; i < words; ++i) {
        uint32_t w = 0;
        for (int n = 0; n < 8; ++n) w |= uint32_t(qd(rng)) << (4 * n);
        hw[i] = w;
    }
    std::vector<half> hs(ngroups), hz(ngroups), hx(K);
    for (size_t i = 0; i < ngroups; ++i) {
        hs[i] = __float2half(sd(rng));
        hz[i] = __float2half(float(zd(rng)));
    }
    std::vector<float> xf(K);
    bench::fill_random(xf, 99, -1.f, 1.f);
    for (int i = 0; i < K; ++i) hx[i] = __float2half(xf[i]);

    uint32_t *d_src, *d_packed;
    half *d_s, *d_z, *d_x;
    float *d_y, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_src, words * 4));
    CUDA_CHECK(cudaMalloc(&d_packed, words * 4));
    CUDA_CHECK(cudaMalloc(&d_s, ngroups * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_z, ngroups * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, size_t(K) * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, size_t(M) * 4));
    CUDA_CHECK(cudaMalloc(&d_ref, size_t(M) * 4));
    CUDA_CHECK(cudaMemcpy(d_src, hw.data(), words * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s, hs.data(), ngroups * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, hz.data(), ngroups * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), size_t(K) * sizeof(half), cudaMemcpyHostToDevice));

    ref_gemv<<<M, 256>>>(d_src, d_s, d_z, d_x, d_ref, M, K);
    CUDA_CHECK_LAUNCH();

    launch_repack(d_src, d_packed, M, K);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemset(d_y, 0, size_t(M) * 4));
    launch_gemv(d_packed, d_s, d_z, d_x, d_y, M, K);
    CUDA_CHECK_LAUNCH();

    std::vector<float> got(M), want(M);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, size_t(M) * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(want.data(), d_ref, size_t(M) * 4, cudaMemcpyDeviceToHost));
    gate.correctness(bench::compare(got, want, 2e-2), 2e-2);

    double ms = bench::time_ms(
        [&] { launch_gemv(d_packed, d_s, d_z, d_x, d_y, M, K); }, 10, 50, true);

    size_t bytes = words * 4                       // packed weights
                   + ngroups * 2 * sizeof(half)    // scales + zeros
                   + size_t(K) * sizeof(half)      // x
                   + size_t(M) * 4;                // y
    double achieved = bench::gbs(bytes, ms);

    std::printf("\nshape          : M=%d K=%d, group=%d\n", M, K, GROUP);
    std::printf("packed weights : %.1f MB\n", words * 4 / 1e6);
    std::printf("scales + zeros : %.1f MB  (%.1f%% of weight bytes)\n",
                ngroups * 4 / 1e6, 100.0 * (ngroups * 4.0) / (words * 4.0));
    std::printf("fp16 would be  : %.1f MB  (%.2fx more traffic)\n",
                double(M) * K * 2 / 1e6, (double(M) * K * 2) / double(bytes));
    std::printf("total bytes    : %.1f MB\n", bytes / 1e6);
    std::printf("time           : %.4f ms\n", ms);

    gate.bandwidth(achieved, 0.80);

    double model_bytes = 7e9 / 2.0 + (7e9 / GROUP) * 4.0;  // 7B int4 + metadata
    std::printf("\nAt this rate a 7B int4 model decodes at %.0f tokens/s;\n",
                (achieved * 1e9) / model_bytes);
    std::printf("in fp16 the same GPU would cap at %.0f tokens/s.\n",
                (bench::copy_ceiling_gbs() * 1e9) / 14e9);

    CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_s)); CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_ref));
    return gate.finish();
}
