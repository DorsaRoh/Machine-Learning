// bench.cuh - shared harness for every exercise in the Kernels masterclass.
//
// You never need to edit this file. You do need to understand it, because it
// decides whether your kernel passes. Read it once before Exercise 02.
//
// What it gives you:
//   CUDA_CHECK(expr)          crash loudly on the first CUDA error
//   bench::device_info()      what GPU you actually rented
//   bench::time_ms(fn)        honest timing: warmup + L2 flush + CUDA events
//   bench::copy_ceiling_gbs() measured, not datasheet, bandwidth ceiling
//   bench::Gate               correctness + performance verdict
//
// Everything here is header-only so a single `nvcc main.cu` builds an exercise.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
//
// CUDA is asynchronous. A kernel launch returns immediately, so an error from
// *inside* the kernel surfaces on some later, unrelated API call. That is why
// every exercise calls CUDA_CHECK(cudaDeviceSynchronize()) after a launch:
// synchronising is what forces the error to come home.

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "\n[CUDA ERROR] %s:%d\n  %s\n  -> %s\n",    \
                         __FILE__, __LINE__, #expr,                          \
                         cudaGetErrorString(_err));                          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// Use immediately after a kernel launch. The first check catches bad launch
// configs (too many threads, too much shared memory); the sync catches faults
// raised by the running kernel itself.
#define CUDA_CHECK_LAUNCH()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)

namespace bench {

// ---------------------------------------------------------------------------
// Device probe
// ---------------------------------------------------------------------------

struct DeviceInfo {
    std::string name;
    int major = 0, minor = 0;
    int sm_count = 0;
    int max_threads_per_sm = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    int regs_per_sm = 0;
    int regs_per_block = 0;
    size_t smem_per_sm = 0;
    size_t smem_per_block = 0;        // default limit, opt-in can raise it
    size_t smem_per_block_optin = 0;  // via cudaFuncAttributeMaxDynamicSharedMemorySize
    size_t l2_bytes = 0;
    size_t global_bytes = 0;
    int mem_bus_width_bits = 0;
    int mem_clock_khz = 0;
    int core_clock_khz = 0;

    // Datasheet peak: bus width * memory clock * 2 (DDR). Real kernels reach
    // 70-90% of this. Treat it as a ceiling you approach, never as a target.
    double peak_bw_gbs() const {
        return 2.0 * mem_clock_khz * 1e3 * (mem_bus_width_bits / 8.0) / 1e9;
    }
};

inline const DeviceInfo& device_info() {
    static DeviceInfo info = [] {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

        DeviceInfo d;
        d.name = p.name;
        d.major = p.major;
        d.minor = p.minor;
        d.sm_count = p.multiProcessorCount;
        d.max_threads_per_sm = p.maxThreadsPerMultiProcessor;
        d.max_threads_per_block = p.maxThreadsPerBlock;
        d.warp_size = p.warpSize;
        d.regs_per_sm = p.regsPerMultiprocessor;
        d.regs_per_block = p.regsPerBlock;
        d.smem_per_sm = p.sharedMemPerMultiprocessor;
        d.smem_per_block = p.sharedMemPerBlock;
        d.smem_per_block_optin = p.sharedMemPerBlockOptin;
        d.l2_bytes = static_cast<size_t>(p.l2CacheSize);
        d.global_bytes = p.totalGlobalMem;
        d.mem_bus_width_bits = p.memoryBusWidth;
        d.mem_clock_khz = p.memoryClockRate;
        d.core_clock_khz = p.clockRate;
        return d;
    }();
    return info;
}

inline void print_device() {
    const DeviceInfo& d = device_info();
    std::printf("GPU            : %s (sm_%d%d)\n", d.name.c_str(), d.major, d.minor);
    std::printf("SMs            : %d   warp = %d threads\n", d.sm_count, d.warp_size);
    std::printf("Threads/SM     : %d   (max resident)\n", d.max_threads_per_sm);
    std::printf("Registers/SM   : %d\n", d.regs_per_sm);
    std::printf("Shared/SM      : %.0f KB   (per block: %.0f KB, opt-in %.0f KB)\n",
                d.smem_per_sm / 1024.0, d.smem_per_block / 1024.0,
                d.smem_per_block_optin / 1024.0);
    std::printf("L2             : %.1f MB\n", d.l2_bytes / (1024.0 * 1024.0));
    std::printf("HBM            : %.1f GB, %d-bit bus, peak %.0f GB/s (datasheet)\n",
                d.global_bytes / 1e9, d.mem_bus_width_bits, d.peak_bw_gbs());
}

// ---------------------------------------------------------------------------
// L2 flush
// ---------------------------------------------------------------------------
//
// If you time the same kernel 100 times on the same small buffer, run 2..100
// read from L2 and you will "discover" 8 TB/s of bandwidth. That number is a
// lie about your kernel's behaviour inside a real model. Between timed
// iterations we scribble over a buffer twice the size of L2 to evict it.

class L2Flusher {
   public:
    L2Flusher() {
        size_t n = std::max<size_t>(device_info().l2_bytes * 2, size_t(1) << 20);
        bytes_ = n;
        CUDA_CHECK(cudaMalloc(&buf_, bytes_));
    }
    ~L2Flusher() { cudaFree(buf_); }
    void flush(cudaStream_t s = 0) const {
        CUDA_CHECK(cudaMemsetAsync(buf_, 0, bytes_, s));
    }

   private:
    void* buf_ = nullptr;
    size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
//
// Rules encoded here, all of which people get wrong:
//   1. Never use wall-clock around an async launch. Use CUDA events.
//   2. Warm up. The first launch pays JIT, context and page-fault costs.
//   3. Flush L2 between iterations (see above).
//   4. Report the MEDIAN, not the mean. One descheduled iteration should not
//      dominate your result; but the best-of-N hides real variance, so median.

struct Timing {
    double median_ms = 0, min_ms = 0, max_ms = 0;
};

// One flusher for the whole process. It owns a buffer twice the size of L2, so
// it must not be instantiated per template specialisation.
inline const L2Flusher& flusher() {
    static L2Flusher f;
    return f;
}

template <typename F>
inline Timing time_ms_full(F&& fn, int warmup = 10, int iters = 100,
                           bool flush_l2 = true) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        if (flush_l2) flusher().flush();
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    Timing t;
    t.median_ms = samples[samples.size() / 2];
    t.min_ms = samples.front();
    t.max_ms = samples.back();
    return t;
}

template <typename F>
inline double time_ms(F&& fn, int warmup = 10, int iters = 100,
                      bool flush_l2 = true) {
    return time_ms_full(std::forward<F>(fn), warmup, iters, flush_l2).median_ms;
}

inline double gbs(size_t bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double gflops(double flops, double ms) { return flops / (ms * 1e-3) / 1e9; }
inline double tflops(double flops, double ms) { return flops / (ms * 1e-3) / 1e12; }

// ---------------------------------------------------------------------------
// Measured bandwidth ceiling
// ---------------------------------------------------------------------------
//
// Performance gates in this course are stated as a fraction of what YOUR GPU
// can actually do, not of a datasheet number, so the same bar is meaningful on
// a T4, an A100 and an H100. The ceiling is measured with the simplest
// possible streaming kernel: read float4, write float4, perfectly coalesced.
// If your kernel is memory-bound, this is the speed of light for it.

namespace detail {
__global__ void copy_f4(const float4* __restrict__ in, float4* __restrict__ out,
                        size_t n4) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < n4; i += stride) out[i] = in[i];
}
}  // namespace detail

inline double copy_ceiling_gbs() {
    static double cached = [] {
        size_t bytes = size_t(256) << 20;  // 256 MB in, 256 MB out
        size_t n4 = bytes / sizeof(float4);
        float4 *in = nullptr, *out = nullptr;
        CUDA_CHECK(cudaMalloc(&in, bytes));
        CUDA_CHECK(cudaMalloc(&out, bytes));
        CUDA_CHECK(cudaMemset(in, 1, bytes));

        int block = 256;
        int grid = device_info().sm_count * 32;
        double ms = time_ms(
            [&] { detail::copy_f4<<<grid, block>>>(in, out, n4); }, 5, 30);
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out));
        return gbs(2 * bytes, ms);
    }();
    return cached;
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

inline void fill_random(std::vector<float>& v, unsigned seed = 1234,
                        float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

struct ErrorStats {
    double max_abs = 0;
    double max_rel = 0;
    size_t first_bad = static_cast<size_t>(-1);
    double got = 0, want = 0;
    size_t nan_count = 0;
};

inline ErrorStats compare(const std::vector<float>& got,
                          const std::vector<float>& want, double tol = 1e-4) {
    ErrorStats s;
    size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(got[i]) || std::isinf(got[i])) {
            ++s.nan_count;
            if (s.first_bad == static_cast<size_t>(-1)) {
                s.first_bad = i;
                s.got = got[i];
                s.want = want[i];
            }
            continue;
        }
        double abs_err = std::fabs(double(got[i]) - double(want[i]));
        double rel_err = abs_err / std::max(1e-6, std::fabs(double(want[i])));
        if (abs_err > s.max_abs) s.max_abs = abs_err;
        if (rel_err > s.max_rel) s.max_rel = rel_err;
        bool bad = abs_err > tol && rel_err > tol;
        if (bad && s.first_bad == static_cast<size_t>(-1)) {
            s.first_bad = i;
            s.got = got[i];
            s.want = want[i];
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------
//
// Two bars. Correctness is binary. Performance is a fraction of the measured
// ceiling (for memory-bound work) or of a provided baseline such as cuBLAS
// (for compute-bound work). The gate prints how far off you are and nothing
// about how to fix it. That part is your job.

class Gate {
   public:
    explicit Gate(std::string exercise) : name_(std::move(exercise)) {
        std::printf("\n=== %s ===\n", name_.c_str());
        print_device();
        std::printf("---------------------------------------------\n");
    }

    void correctness(const ErrorStats& s, double tol = 1e-4) {
        bool ok = s.nan_count == 0 && s.first_bad == static_cast<size_t>(-1);
        correct_ = ok;
        std::printf("correctness    : %s  (max_abs %.3g, max_rel %.3g, tol %.3g)\n",
                    ok ? "PASS" : "FAIL", s.max_abs, s.max_rel, tol);
        if (!ok) {
            if (s.nan_count)
                std::printf("                 %zu NaN/Inf values produced\n", s.nan_count);
            std::printf("                 first mismatch at index %zu: got %.6g, want %.6g\n",
                        s.first_bad, s.got, s.want);
        }
    }

    // Memory-bound gate: fraction of the measured streaming-copy ceiling.
    void bandwidth(double achieved_gbs, double min_fraction) {
        double ceil_gbs = copy_ceiling_gbs();
        double frac = achieved_gbs / ceil_gbs;
        perf_ = frac >= min_fraction;
        std::printf("bandwidth      : %.1f GB/s  (%.0f%% of measured ceiling %.1f GB/s)\n",
                    achieved_gbs, 100.0 * frac, ceil_gbs);
        std::printf("perf gate      : %s  (need >= %.0f%%)\n",
                    perf_ ? "PASS" : "FAIL", 100.0 * min_fraction);
    }

    // Compute-bound gate: fraction of a reference implementation's throughput.
    void relative(double achieved, double baseline, double min_fraction,
                  const char* unit, const char* baseline_name) {
        double frac = achieved / baseline;
        perf_ = frac >= min_fraction;
        std::printf("throughput     : %.2f %s  (%.0f%% of %s at %.2f %s)\n", achieved,
                    unit, 100.0 * frac, baseline_name, baseline, unit);
        std::printf("perf gate      : %s  (need >= %.0f%%)\n",
                    perf_ ? "PASS" : "FAIL", 100.0 * min_fraction);
    }

    // Speedup gate: must be at least N times faster than a given time.
    void speedup(double baseline_ms, double achieved_ms, double min_speedup,
                 const char* baseline_name) {
        double s = baseline_ms / achieved_ms;
        perf_ = s >= min_speedup;
        std::printf("time           : %.3f ms vs %s %.3f ms  (%.2fx)\n", achieved_ms,
                    baseline_name, baseline_ms, s);
        std::printf("perf gate      : %s  (need >= %.2fx)\n", perf_ ? "PASS" : "FAIL",
                    min_speedup);
    }

    void no_perf_gate() { perf_ = true; }

    int finish() const {
        bool ok = correct_ && perf_;
        std::printf("---------------------------------------------\n");
        std::printf("%s: %s\n\n", name_.c_str(), ok ? "PASSED" : "NOT YET");
        return ok ? 0 : 1;
    }

   private:
    std::string name_;
    bool correct_ = false;
    bool perf_ = false;
};

}  // namespace bench
