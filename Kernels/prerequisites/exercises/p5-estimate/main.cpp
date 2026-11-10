// P5 - estimate
//
// The last prerequisite, and the one that turns into Chapter 01 unchanged.
//
// Part 1 is arithmetic you do without a computer. Part 2 makes you commit to a
// prediction in writing. Part 3 measures. The gap between parts 2 and 3 is the
// only honest measure of whether you understand a machine.
//
// Chapter: P8

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"

// ================================================================ part 1
// Counting. All of these are one line.

// c[i] = a[i] + b[i], fp32.
long long bytes_vector_add(long long n) {
    // TODO
    (void)n;
    return 0;
}
long long flops_vector_add(long long n) {
    // TODO
    (void)n;
    return 0;
}

// y[i] = alpha * x[i] + y[i], fp32. Careful: y is both read and written.
long long bytes_saxpy(long long n) {
    // TODO
    (void)n;
    return 0;
}
long long flops_saxpy(long long n) {
    // TODO
    (void)n;
    return 0;
}

// sum += a[i] * b[i], fp32. The output is a single scalar; ignore it.
long long bytes_dot(long long n) {
    // TODO
    (void)n;
    return 0;
}
long long flops_dot(long long n) {
    // TODO
    (void)n;
    return 0;
}

// C(MxN) = A(MxK) @ B(KxN), fp32, each element touched once.
long long bytes_matmul(long long M, long long N, long long K) {
    // TODO
    (void)M; (void)N; (void)K;
    return 0;
}
long long flops_matmul(long long M, long long N, long long K) {
    // TODO
    (void)M; (void)N; (void)K;
    return 0;
}

// ================================================================ part 2
// The model.

// FLOPs per byte.
double arithmetic_intensity(long long flops, long long bytes) {
    // TODO
    (void)flops; (void)bytes;
    return 0.0;
}

// The intensity at which a machine is perfectly balanced.
// Units: GFLOP/s and GB/s in, FLOP/byte out.
double ridge_point(double peak_gflops, double peak_gbps) {
    // TODO
    (void)peak_gflops; (void)peak_gbps;
    return 0.0;
}

// Is this operation limited by memory on this machine?
bool is_memory_bound(double intensity, double ridge) {
    // TODO
    (void)intensity; (void)ridge;
    return false;
}

// Predicted milliseconds for a memory-bound operation.
double predicted_ms(long long bytes, double gbps) {
    // TODO: watch the units. GB/s is 1e9 bytes per second, and the answer is
    // in milliseconds.
    (void)bytes; (void)gbps;
    return 0.0;
}

// Working backwards: what bandwidth did a run actually achieve?
double achieved_gbps(long long bytes, double ms) {
    // TODO
    (void)bytes; (void)ms;
    return 0.0;
}

// ================================================================ part 3
// Write down your predictions BEFORE you run anything. These two constants
// are not graded -- the harness just prints how close you were, which is
// between you and your conscience.

const double MY_PREDICTED_BANDWIDTH_GBPS = 0.0;  // TODO: your machine's
const double MY_PREDICTED_FUSION_SPEEDUP = 0.0;  // TODO: 3 passes -> 1 pass

// ================================================================ part 4
// The code to measure. Implement these; the harness times them.

// The bandwidth probe: read n floats, write n floats, do no arithmetic at
// all. Nothing can be faster than this, which is why the main course calls it
// the "copy ceiling" and gates `02-vector-add` against it.
void copy_floats(const float* src, float* dst, long long n) {
    // TODO
    (void)src; (void)dst; (void)n;
}

double sum_floats(const float* a, long long n) {
    // TODO: pure streaming read. Accumulate in double.
    (void)a; (void)n;
    return 0.0;
}

void vector_add(const float* a, const float* b, float* c, long long n) {
    // TODO
    (void)a; (void)b; (void)c; (void)n;
}

// Three separate passes, each reading and writing a whole array:
//   b = a * 2 ;  c = b + 1 ;  d = c * c
void three_passes(const float* a, float* b, float* c, float* d, long long n) {
    // TODO: three separate loops. Yes, really three.
    (void)a; (void)b; (void)c; (void)d; (void)n;
}

// The same arithmetic in one pass, reading a and writing d only.
void fused(const float* a, float* d, long long n) {
    // TODO
    (void)a; (void)d; (void)n;
}

// ------------------------------------------------------------------------

template <typename F>
static double time_ms(F f) {
    f();  // warm up: the first touch of a fresh page is not what you want to
          // measure, a lesson that recurs in Chapter 17
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    check::Suite s("p5-estimate");

    // ---- part 1 ----------------------------------------------------------
    const long long M = 1000000;
    s.eq_int("bytes_vector_add(1e6)", bytes_vector_add(M), 12000000LL);
    s.eq_int("flops_vector_add(1e6)", flops_vector_add(M), 1000000LL);
    s.eq_int("bytes_saxpy(1e6)", bytes_saxpy(M), 12000000LL);
    s.eq_int("flops_saxpy(1e6)", flops_saxpy(M), 2000000LL);
    s.eq_int("bytes_dot(1e6)", bytes_dot(M), 8000000LL);
    s.eq_int("flops_dot(1e6)", flops_dot(M), 2000000LL);
    s.eq_int("bytes_matmul(1024)", bytes_matmul(1024, 1024, 1024), 12582912LL);
    s.eq_int("flops_matmul(1024)", flops_matmul(1024, 1024, 1024), 2147483648LL);
    s.eq_int("flops_matmul(8192) -- an int would have overflowed twice over",
             flops_matmul(8192, 8192, 8192), 1099511627776LL);

    // ---- part 2 ----------------------------------------------------------
    double ai_add = arithmetic_intensity(flops_vector_add(M), bytes_vector_add(M));
    double ai_dot = arithmetic_intensity(flops_dot(M), bytes_dot(M));
    double ai_mm = arithmetic_intensity(flops_matmul(1024, 1024, 1024),
                                        bytes_matmul(1024, 1024, 1024));
    s.eq_double("intensity of vector add", ai_add, 0.0833333333, 1e-6);
    s.eq_double("intensity of dot", ai_dot, 0.25, 1e-9);
    s.eq_double("intensity of matmul 1024", ai_mm, 170.6666667, 1e-5);

    // An A100: 1555 GB/s, 19.5 TFLOP/s fp32, 312 TFLOP/s on tensor cores.
    double ridge_fp32 = ridge_point(19500.0, 1555.0);
    double ridge_tc = ridge_point(312000.0, 1555.0);
    s.eq_double("A100 ridge point, fp32", ridge_fp32, 12.5401929, 1e-5);
    s.eq_double("A100 ridge point, tensor cores", ridge_tc, 200.6430868, 1e-5);

    s.ok("vector add is memory-bound on an A100",
         is_memory_bound(ai_add, ridge_fp32));
    s.ok("dot product is memory-bound on an A100",
         is_memory_bound(ai_dot, ridge_fp32));
    s.ok("matmul 1024 is COMPUTE-bound on A100 fp32",
         !is_memory_bound(ai_mm, ridge_fp32));
    s.ok("the same matmul is MEMORY-bound once tensor cores are switched on",
         is_memory_bound(ai_mm, ridge_tc));

    s.eq_double("predicted_ms(120 MB, 50 GB/s)", predicted_ms(120000000LL, 50.0),
                2.4, 1e-9);
    s.eq_double("achieved_gbps(120 MB in 3 ms)", achieved_gbps(120000000LL, 3.0),
                40.0, 1e-9);
    s.eq_double("round trip: predict then invert",
                achieved_gbps(bytes_vector_add(M),
                              predicted_ms(bytes_vector_add(M), 37.5)),
                37.5, 1e-6);

    // ---- parts 3 and 4 ---------------------------------------------------
    const long long N = 20000000;  // 80 MB per array
    std::vector<float> a(N), b(N, 0.0f), c(N, 1.0f), d(N, 0.0f), e(N, 0.0f);
    for (long long i = 0; i < N; i++) a[i] = float(i % 17) * 0.25f;

    double t_copy = time_ms([&] { copy_floats(a.data(), d.data(), N); });
    double measured_gbps = achieved_gbps(8 * N, t_copy);
    s.ok("copy_floats actually copied", d[11] == a[11] && d[N - 1] == a[N - 1]);

    // `sink` exists so the compiler cannot delete sum_floats as dead code.
    // Benchmarking something the optimiser removed is the oldest mistake in
    // performance work, and you will meet it again in Chapter 16.
    double sink = 0.0;
    double t_read = time_ms([&] { sink += sum_floats(a.data(), N); });
    s.ok("sum_floats actually read the array", sink != 0.0);

    double t_add = time_ms([&] { vector_add(a.data(), c.data(), b.data(), N); });
    s.eq_double("vector_add is correct", b[7], double(a[7]) + 1.0, 1e-6);

    double t_three = time_ms([&] {
        three_passes(a.data(), b.data(), c.data(), d.data(), N);
    });
    double t_fused = time_ms([&] { fused(a.data(), e.data(), N); });

    bool agree = true;
    for (long long i = 0; i < N && agree; i += 7919)
        agree = std::fabs(double(d[i]) - double(e[i])) < 1e-3;
    s.ok("fused and three-pass produce identical results", agree);
    s.ok("both versions actually ran", d[3] != 0.0f && e[3] != 0.0f);

    double predicted_add = predicted_ms(bytes_vector_add(N), measured_gbps);
    double speedup = t_three / (t_fused > 0 ? t_fused : 1e9);

    std::printf("\n  --- measured ----------------------------------------\n");
    std::printf("  copy, 160 MB moved        : %7.2f ms  -> %6.1f GB/s  <- ceiling\n",
                t_copy, measured_gbps);
    std::printf("  vector add, 240 MB moved  : %7.2f ms  -> %6.1f GB/s\n", t_add,
                achieved_gbps(bytes_vector_add(N), t_add));
    std::printf("  sum,  80 MB read          : %7.2f ms  -> %6.1f GB/s  <- see note\n",
                t_read, achieved_gbps(4 * N, t_read));
    std::printf("  three separate passes     : %7.2f ms\n", t_three);
    std::printf("  one fused pass            : %7.2f ms\n", t_fused);
    std::printf("\n  --- predicted vs actual -----------------------------\n");
    std::printf("  bandwidth : you said %6.1f GB/s, machine says %6.1f GB/s\n",
                MY_PREDICTED_BANDWIDTH_GBPS, measured_gbps);
    std::printf("  fusion    : you said %6.2fx,     machine says %6.2fx\n",
                MY_PREDICTED_FUSION_SPEEDUP, speedup);
    std::printf("  vector add: model says %6.2f ms, machine says %6.2f ms\n\n",
                predicted_add, t_add);

    s.ok("fusing three passes into one is at least 1.5x faster",
         speedup >= 1.5);
    s.ok("the bandwidth model predicts vector_add within a factor of 3",
         t_add > 0 && predicted_add > 0 &&
             std::max(predicted_add / t_add, t_add / predicted_add) < 3.0);

    std::printf("  Note: `sum` is almost certainly far below the ceiling, and\n");
    std::printf("  not because of memory. Work out why before reading the\n");
    std::printf("  README -- the answer is the subject of Chapter 07.\n");

    s.note("Chapter 01 is this file with a GPU's numbers substituted in.");
    return s.finish();
}
