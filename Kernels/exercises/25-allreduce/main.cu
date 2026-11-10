// Harness for 25-allreduce. Do not edit.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "bench.cuh"
#include "kernel.cu"

#ifdef HAVE_NCCL
#include <nccl.h>
#define NCCL_CHECK(expr)                                                     \
    do {                                                                     \
        ncclResult_t r_ = (expr);                                            \
        if (r_ != ncclSuccess) {                                             \
            std::fprintf(stderr, "NCCL error %s at %s:%d\n",                 \
                         ncclGetErrorString(r_), __FILE__, __LINE__);        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
#endif

static float value_of(int g, long long i) {
    return float((g + 1) * int(i % 7 + 1));  // small integers: sums are exact
}

int main(int argc, char** argv) {
    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (argc > 1) {
        int want = std::atoi(argv[1]);
        if (want > 0 && want < ngpu) ngpu = want;
    }
    if (ngpu < 2) {
        std::printf(
            "\n25-allreduce needs at least 2 GPUs; this machine has %d.\n"
            "Rent a 2x or 4x instance for an hour (see setup/GETTING_STARTED.md).\n"
            "Chapter 25's pencil exercises 25.1, 25.2 and 25.4 are worth doing\n"
            "regardless, and 25.1 in particular needs real hardware.\n\n",
            ngpu);
        return 0;
    }

    std::printf("\ntopology (peer access):\n     ");
    for (int j = 0; j < ngpu; ++j) std::printf(" gpu%d", j);
    std::printf("\n");
    for (int i = 0; i < ngpu; ++i) {
        std::printf("gpu%d ", i);
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < ngpu; ++j) {
            if (i == j) { std::printf("    -"); continue; }
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    CUDA_CHECK(e);
                cudaGetLastError();
            }
            std::printf("  %s", can ? " p2p" : " pci");
        }
        std::printf("\n");
    }
    std::printf("(p2p = direct peer access; pci = staged through the host)\n");

    bench::Gate gate("25-allreduce");

    std::vector<float*> bufs(ngpu), scratch(ngpu), nbuf(ngpu);
    std::vector<cudaStream_t> streams(ngpu);
    const long long nmax = 1LL << 24;  // 64 MB
    for (int g = 0; g < ngpu; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&bufs[g], nmax * 4));
        CUDA_CHECK(cudaMalloc(&scratch[g], nmax * 4));
        CUDA_CHECK(cudaMalloc(&nbuf[g], nmax * 4));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
    }

    auto fill = [&](long long n) {
        std::vector<float> h(n);
        for (int g = 0; g < ngpu; ++g) {
            for (long long i = 0; i < n; ++i) h[i] = value_of(g, i);
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(bufs[g], h.data(), n * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(nbuf[g], h.data(), n * 4, cudaMemcpyHostToDevice));
        }
    };
    auto sync_all = [&] {
        for (int g = 0; g < ngpu; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };
    auto timeit = [&](auto&& fn, int warmup, int iters) {
        for (int i = 0; i < warmup; ++i) fn();
        sync_all();
        std::vector<double> ts;
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            sync_all();
            auto t1 = std::chrono::high_resolution_clock::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        return ts[ts.size() / 2];
    };

    // ---- correctness ------------------------------------------------------
    const long long ncheck = 1 << 20;
    fill(ncheck);
    ring_allreduce(bufs.data(), scratch.data(), int(ncheck), ngpu, streams.data());
    sync_all();

    std::vector<float> want(ncheck, 0.f);
    for (long long i = 0; i < ncheck; ++i)
        for (int g = 0; g < ngpu; ++g) want[i] += value_of(g, i);

    bool all_ok = true;
    bench::ErrorStats worst;
    std::vector<float> got(ncheck);
    std::printf("\n");
    for (int g = 0; g < ngpu; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(got.data(), bufs[g], ncheck * 4, cudaMemcpyDeviceToHost));
        bench::ErrorStats s = bench::compare(got, want, 0.0);
        bool ok = s.first_bad == static_cast<size_t>(-1) && s.nan_count == 0;
        std::printf("gpu%d result : %s\n", g, ok ? "ok" : "WRONG");
        if (!ok) { all_ok = false; worst = s; }
    }
    gate.correctness(all_ok ? bench::ErrorStats{} : worst, 0.0);

    // ---- speed ------------------------------------------------------------
#ifdef HAVE_NCCL
    std::vector<ncclComm_t> comms(ngpu);
    std::vector<int> devs(ngpu);
    for (int g = 0; g < ngpu; ++g) devs[g] = g;
    NCCL_CHECK(ncclCommInitAll(comms.data(), ngpu, devs.data()));
#endif

    std::printf("\n%12s %12s %12s %10s %10s\n", "bytes", "yours ms", "nccl ms",
                "yours GB/s", "nccl GB/s");
    double mine_at_max = 0, nccl_at_max = 0;
    for (long long n = 1 << 10; n <= nmax; n <<= 2) {
        fill(n);
        double mine = timeit(
            [&] {
                ring_allreduce(bufs.data(), scratch.data(), int(n), ngpu,
                               streams.data());
            },
            3, 15);
        double nccl_ms = 0;
#ifdef HAVE_NCCL
        nccl_ms = timeit(
            [&] {
                NCCL_CHECK(ncclGroupStart());
                for (int g = 0; g < ngpu; ++g)
                    NCCL_CHECK(ncclAllReduce(nbuf[g], nbuf[g], size_t(n), ncclFloat,
                                             ncclSum, comms[g], streams[g]));
                NCCL_CHECK(ncclGroupEnd());
            },
            3, 15);
#endif
        // algorithm bandwidth: bytes the user cares about, per unit time
        double gbs_mine = double(n) * 4 / (mine * 1e-3) / 1e9;
        double gbs_nccl = nccl_ms > 0 ? double(n) * 4 / (nccl_ms * 1e-3) / 1e9 : 0;
        std::printf("%12lld %12.4f %12.4f %10.2f %10.2f\n", n * 4, mine, nccl_ms,
                    gbs_mine, gbs_nccl);
        if (n == nmax) { mine_at_max = gbs_mine; nccl_at_max = gbs_nccl; }
    }

    double ring_bytes = 2.0 * (ngpu - 1) / ngpu * double(nmax) * 4;
    std::printf("\nring moves %.0f MB per GPU for a %.0f MB buffer (%d GPUs).\n",
                ring_bytes / 1e6, double(nmax) * 4 / 1e6, ngpu);

#ifdef HAVE_NCCL
    gate.relative(mine_at_max, nccl_at_max, 0.60, "GB/s", "NCCL");
    for (int g = 0; g < ngpu; ++g) NCCL_CHECK(ncclCommDestroy(comms[g]));
#else
    std::printf("\nNo NCCL baseline available, so the performance gate is skipped.\n");
    std::printf("Install NCCL and rerun before calling this finished.\n");
    gate.no_perf_gate();
#endif

    return gate.finish();
}
