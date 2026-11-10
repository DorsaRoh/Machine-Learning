// Harness for 18-race-hunt. Do not edit.
//
// Runs each kernel many times and reports disagreement with a CPU reference
// and bit-level reproducibility across runs. A race that fires one run in a
// hundred still fails here, which is the point. A race that fires zero times
// today still fails `make sanitize`, which is also the point.

#include <cstdio>
#include <cstring>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

int main(int argc, char** argv) {
    int runs = (argc > 1) ? std::atoi(argv[1]) : 500;
    const int n = 1 << 20;
    const int tiles = 4;
    const int groups = 8;
    const int blocks = n / N_PER_BLOCK;              // cases 2, 3, 4
    const int blocks1 = n / (N_PER_BLOCK * tiles);   // case 1

    bench::Gate gate("18-race-hunt");

    std::vector<float> h_in(n);
    bench::fill_random(h_in, 81, 0.0f, 1.0f);

    float *d_in, *d_out, *d_scale;
    int* d_ready;
    CUDA_CHECK(cudaMalloc(&d_in, size_t(n) * 4));
    CUDA_CHECK(cudaMalloc(&d_out, size_t(n) * 4));
    CUDA_CHECK(cudaMalloc(&d_scale, 4));
    CUDA_CHECK(cudaMalloc(&d_ready, 4));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size_t(n) * 4, cudaMemcpyHostToDevice));

    // ---- CPU references ---------------------------------------------------
    std::vector<float> ref1(n), ref2(groups, 0.0f), ref3(blocks), ref4(n);
    for (int g = 0; g < n; ++g) {
        double a = 0;
        for (int k = 0; k < 4; ++k) a += (g + k < n) ? double(h_in[g + k]) : 0.0;
        ref1[g] = float(a);
    }
    for (int b = 0; b < blocks; ++b) {
        double s = 0;
        for (int x = 0; x < N_PER_BLOCK; ++x) {
            long long i = (long long)b * N_PER_BLOCK + x;
            s += double((int)((i * 37) % 11));
        }
        ref2[b % groups] += float(s);
        double s3 = 0;
        for (int x = 0; x < N_PER_BLOCK; ++x) s3 += double(h_in[b * N_PER_BLOCK + x]);
        ref3[b] = float(s3);
    }
    for (int i = 0; i < n; ++i) ref4[i] = h_in[i] * 2.0f;

    struct Case {
        const char* name;
        int out_count;
        const std::vector<float>* ref;
        double tol;
    };
    Case cases[4] = {{"1 sliding window ", n, &ref1, 1e-5},
                     {"2 shared counter ", groups, &ref2, 0.0},
                     {"3 warp unroll    ", blocks, &ref3, 1e-4},
                     {"4 publish / wait ", n, &ref4, 0.0}};

    bool all_pass = true;
    bench::ErrorStats worst;
    worst.first_bad = 0;

    std::printf("\n case              | buggy: bad runs | fixed: bad runs | fixed: stable?\n");
    std::printf("-------------------+-----------------+-----------------+---------------\n");

    for (int c = 0; c < 4; ++c) {
        int bad_runs[2] = {0, 0};
        bool stable[2] = {true, true};
        std::vector<float> first[2];

        for (int which = 0; which < 2; ++which) {
            for (int r = 0; r < runs; ++r) {
                CUDA_CHECK(cudaMemset(d_out, 0, size_t(n) * 4));
                CUDA_CHECK(cudaMemset(d_ready, 0, 4));
                CUDA_CHECK(cudaMemset(d_scale, 0, 4));
                switch (c) {
                    case 0:
                        if (!which) bug_1<<<blocks1, N_PER_BLOCK>>>(d_in, d_out, n, tiles);
                        else        fix_1<<<blocks1, N_PER_BLOCK>>>(d_in, d_out, n, tiles);
                        break;
                    case 1:
                        if (!which) bug_2<<<blocks, N_PER_BLOCK>>>(d_out, groups);
                        else        fix_2<<<blocks, N_PER_BLOCK>>>(d_out, groups);
                        break;
                    case 2:
                        if (!which) bug_3<<<blocks, N_PER_BLOCK>>>(d_in, d_out, n);
                        else        fix_3<<<blocks, N_PER_BLOCK>>>(d_in, d_out, n);
                        break;
                    default:
                        if (!which)
                            bug_4<<<blocks, N_PER_BLOCK>>>(d_in, d_out, d_scale, d_ready, n);
                        else
                            fix_4<<<blocks, N_PER_BLOCK>>>(d_in, d_out, d_scale, d_ready, n);
                        break;
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<float> got(cases[c].out_count);
                CUDA_CHECK(cudaMemcpy(got.data(), d_out, size_t(cases[c].out_count) * 4,
                                      cudaMemcpyDeviceToHost));
                bench::ErrorStats st = bench::compare(got, *cases[c].ref, cases[c].tol);
                if (st.first_bad != static_cast<size_t>(-1) || st.nan_count) {
                    ++bad_runs[which];
                    if (which == 1) worst = st;
                }
                if (r == 0) {
                    first[which] = got;
                } else if (std::memcmp(first[which].data(), got.data(),
                                       size_t(cases[c].out_count) * 4) != 0) {
                    stable[which] = false;
                }
            }
        }
        bool pass = bad_runs[1] == 0 && stable[1];
        if (!pass) all_pass = false;
        std::printf(" %s | %15d | %15d | %13s\n", cases[c].name, bad_runs[0],
                    bad_runs[1], stable[1] ? "yes" : "NO");
    }

    std::printf("\n%d runs each. A buggy kernel with 0 bad runs is still buggy --\n", runs);
    std::printf("it just did not lose the race today. Run `make sanitize`.\n");

    gate.correctness(all_pass ? bench::ErrorStats{} : worst, 0.0);
    gate.no_perf_gate();

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaFree(d_ready));
    return gate.finish();
}
