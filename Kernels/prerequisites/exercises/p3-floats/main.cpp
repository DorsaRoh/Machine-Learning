// P3 - floats
//
// Where you find out that `float` is not a real number, that a + b + c is not
// a + (b + c), and that a GPU reduction is therefore allowed to disagree with
// your CPU reference. This is Chapter 15 in miniature, and the reason every
// gate in the main course has a tolerance rather than an equals sign.
//
// Chapter: P4

#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"

// ================================================================ part 1
// Bit arithmetic. You will write these two expressions in nearly every kernel,
// because threads are grouped into warps of 32.

// Which lane within its warp is thread `tid`? Equivalent to tid % 32, but use
// a bitwise AND -- that is what the hardware does and what you will read in
// other people's kernels.
int lane_of(int tid) {
    // TODO
    (void)tid;
    return -1;
}

// Which warp does thread `tid` belong to? Equivalent to tid / 32, using a
// shift.
int warp_of(int tid) {
    // TODO
    (void)tid;
    return -1;
}

// ================================================================ part 2
// Integer limits.

// How many elements in a rows x cols matrix? The answer can exceed what an
// int holds, so the return type does not.
long long num_elements(int rows, int cols) {
    // TODO: careful. Where exactly does the multiplication happen, and in
    // what type?
    (void)rows;
    (void)cols;
    return 0;
}

// Bytes needed to store that matrix as floats.
long long bytes_for_float_matrix(int rows, int cols) {
    // TODO
    (void)rows;
    (void)cols;
    return 0;
}

// ================================================================ part 3
// Where float runs out.

// The smallest positive integer that a float CANNOT represent exactly.
// Find it with a loop; do not look it up and type the constant.
// (Hint on method, not answer: count upward and test whether float(n) and
// float(n+1) are distinguishable.)
double first_unrepresentable_integer() {
    // TODO
    return 0.0;
}

// Machine epsilon: the smallest e such that 1.0f + e is not 1.0f.
// Find it by halving.
double float_epsilon() {
    // TODO
    return 0.0;
}

// Starting from 0.0f, repeatedly add 1.0f. Return the value at which adding
// 1.0f stops changing anything. (Do not actually loop 17 million times unless
// you want to -- you may start the loop near where you expect the answer.)
double where_counting_stalls() {
    // TODO
    return 0.0;
}

// ================================================================ part 4
// Summation order.

// Left to right, accumulating in float.
double sum_naive_float(const float* a, int n) {
    // TODO: the accumulator must be a float. That is the point.
    (void)a;
    (void)n;
    return 0.0;
}

// Left to right, accumulating in double.
double sum_wide(const float* a, int n) {
    // TODO
    (void)a;
    (void)n;
    return 0.0;
}

// Pairwise (tree) summation, float accumulators throughout: split the range in
// half, sum each half, add the two results. This is the shape of Chapter 07's
// parallel reduction, and -- as you are about to measure -- it is also far
// more accurate than the sequential version.
double sum_tree_float(const float* a, int n) {
    // TODO
    (void)a;
    (void)n;
    return 0.0;
}

// ================================================================ part 5
// Cancellation.

// Variance the textbook way: E[x^2] - E[x]^2, every intermediate a float.
// This is the formula that looks elegant and loses all your precision.
double variance_naive_float(const float* a, int n) {
    // TODO
    (void)a;
    (void)n;
    return 0.0;
}

// Variance in two passes: compute the mean, then the mean of (x - mean)^2.
// Accumulate in double. This is what Chapter 10's LayerNorm does, and why.
double variance_two_pass(const float* a, int n) {
    // TODO
    (void)a;
    (void)n;
    return 0.0;
}

// ================================================================ part 6
// Overflow, and the trick that avoids it.

// Softmax, numerically stable: subtract the maximum before exponentiating.
// out[i] = exp(x[i] - max) / sum_j exp(x[j] - max)
void softmax_stable(const float* x, float* out, int n) {
    // TODO
    (void)x;
    (void)out;
    (void)n;
}

// Softmax, done the way the formula is written on the page, with no shift.
// You are implementing this one *so that you can watch it fail.*
void softmax_naive(const float* x, float* out, int n) {
    // TODO
    (void)x;
    (void)out;
    (void)n;
}

// ================================================================ part 7
// NaN. Without using any library function.

bool is_nan(float x) {
    // TODO: one comparison. What is the one value that is not equal to
    // itself?
    (void)x;
    return false;
}

int main() {
    check::Suite s("p3-floats");

    // ---- part 1 ----------------------------------------------------------
    s.eq_int("lane_of(0)", lane_of(0), 0);
    s.eq_int("lane_of(31)", lane_of(31), 31);
    s.eq_int("lane_of(32)", lane_of(32), 0);
    s.eq_int("lane_of(100)", lane_of(100), 4);
    s.eq_int("warp_of(0)", warp_of(0), 0);
    s.eq_int("warp_of(31)", warp_of(31), 0);
    s.eq_int("warp_of(32)", warp_of(32), 1);
    s.eq_int("warp_of(255)", warp_of(255), 7);

    bool agrees = true;
    for (int t = 0; t < 2048 && agrees; t++)
        agrees = (lane_of(t) == t % 32) && (warp_of(t) == t / 32);
    s.ok("lane/warp agree with % and / for 2048 threads", agrees);

    // ---- part 2 ----------------------------------------------------------
    s.eq_int("num_elements(1024, 1024)", num_elements(1024, 1024), 1048576);
    s.eq_int("num_elements(100000, 100000) -- overflows an int",
             num_elements(100000, 100000), 10000000000LL);
    s.eq_int("bytes_for_float_matrix(100000, 100000)",
             bytes_for_float_matrix(100000, 100000), 40000000000LL);

    // ---- part 3 ----------------------------------------------------------
    s.eq_double("first integer a float cannot represent",
                first_unrepresentable_integer(), 16777217.0);
    s.eq_double("float epsilon", float_epsilon(), 1.1920928955078125e-07, 1e-16);
    s.eq_double("where counting by 1 stalls", where_counting_stalls(), 16777216.0);

    // ---- part 4 ----------------------------------------------------------
    const int N = 20000000;
    std::vector<float> ones(N, 1.0f);

    double naive = sum_naive_float(ones.data(), N);
    double wide = sum_wide(ones.data(), N);
    double tree = sum_tree_float(ones.data(), N);

    s.eq_double("sequential float sum of 2e7 ones stalls at 2^24", naive,
                16777216.0);
    s.eq_double("double accumulator gets it right", wide, 20000000.0);
    s.eq_double("tree sum, still only floats, also gets it right", tree,
                20000000.0, 1.0);

    std::printf("\n  sequential float : %.1f\n", naive);
    std::printf("  tree float       : %.1f\n", tree);
    std::printf("  double           : %.1f\n", wide);
    std::printf("  true answer      : %.1f\n\n", 20000000.0);
    std::printf("  Same data. Same arithmetic. Same precision. Different order.\n");
    std::printf("  That is why a GPU reduction and your CPU loop disagree.\n\n");

    // Order matters even on well-behaved data.
    const int H = 1000000;
    std::vector<float> harm(H), harm_rev(H);
    for (int i = 0; i < H; i++) harm[i] = 1.0f / float(i + 1);
    for (int i = 0; i < H; i++) harm_rev[i] = harm[H - 1 - i];
    double fwd = sum_naive_float(harm.data(), H);
    double rev = sum_naive_float(harm_rev.data(), H);
    s.ok("summing largest-first and smallest-first give different answers",
         fwd != rev);
    double truth = sum_wide(harm.data(), H);
    s.ok("smallest-first is the more accurate of the two",
         std::fabs(rev - truth) < std::fabs(fwd - truth));

    // ---- part 5 ----------------------------------------------------------
    // 1000 values spread over a range of 1000, but sitting at 10 million.
    // Population variance of 0..999 is (1000^2 - 1)/12 = 83333.25.
    const int V = 1000;
    std::vector<float> off(V);
    for (int i = 0; i < V; i++) off[i] = 1e7f + float(i);

    double vn = variance_naive_float(off.data(), V);
    double v2 = variance_two_pass(off.data(), V);
    s.eq_double("two-pass variance is correct", v2, 83333.25, 1e-3);
    s.ok("naive E[x^2] - E[x]^2 in float is visibly wrong",
         std::fabs(vn - 83333.25) > 100.0 || is_nan(float(vn)));
    std::printf("  variance, naive float : %.4f\n", vn);
    std::printf("  variance, two-pass    : %.4f\n", v2);
    std::printf("  true                  : %.4f\n\n", 83333.25);

    // Sanity: the same formula is fine when the data is centred.
    std::vector<float> centred(V);
    for (int i = 0; i < V; i++) centred[i] = float(i);
    s.eq_double("two-pass works on centred data too",
                variance_two_pass(centred.data(), V), 83333.25, 1e-6);

    // ---- part 6 ----------------------------------------------------------
    std::vector<float> small = {0.0f, 1.0f, 2.0f}, out(3, 0.0f);
    softmax_stable(small.data(), out.data(), 3);
    s.eq_array("softmax of {0,1,2}", {out[0], out[1], out[2]},
               {0.09003057, 0.24472847, 0.66524096}, 1e-6);

    // The same distribution, shifted up by 1000. Mathematically identical.
    std::vector<float> huge = {1000.0f, 1001.0f, 1002.0f}, outh(3, 0.0f);
    softmax_stable(huge.data(), outh.data(), 3);
    s.eq_array("softmax of {1000,1001,1002} -- same answer",
               {outh[0], outh[1], outh[2]},
               {0.09003057, 0.24472847, 0.66524096}, 1e-6);

    std::vector<float> outn(3, 0.0f);
    softmax_naive(huge.data(), outn.data(), 3);
    s.ok("the unshifted version produces NaN or inf on the same input",
         is_nan(outn[0]) || std::isinf(outn[0]));
    std::printf("  naive softmax of {1000,1001,1002}: %f %f %f\n\n", outn[0],
                outn[1], outn[2]);

    double tot = outh[0] + outh[1] + outh[2];
    s.eq_double("stable softmax sums to 1", tot, 1.0, 1e-6);

    // Hard case: all elements equal and enormous.
    std::vector<float> flat = {500.0f, 500.0f, 500.0f, 500.0f}, outf(4, 0.0f);
    softmax_stable(flat.data(), outf.data(), 4);
    s.eq_array("softmax of four equal values is uniform",
               {outf[0], outf[1], outf[2], outf[3]},
               {0.25, 0.25, 0.25, 0.25}, 1e-6);

    // ---- part 7 ----------------------------------------------------------
    float zero = 0.0f;
    float nan_value = zero / zero;
    s.ok("is_nan(nan)", is_nan(nan_value));
    s.ok("is_nan(0.0f) is false", !is_nan(0.0f));
    s.ok("is_nan(1.0f/0.0f) is false -- infinity is not NaN",
         !is_nan(1.0f / zero));
    s.ok("NaN is contagious: nan + 1 is still NaN", is_nan(nan_value + 1.0f));
    s.ok("even nan * 0 is NaN", is_nan(nan_value * 0.0f));
    s.ok("nan == nan is false", !(nan_value == nan_value));
    s.ok("nan < 1 and nan >= 1 are BOTH false",
         !(nan_value < 1.0f) && !(nan_value >= 1.0f));

    s.note("Every tolerance in the main course exists because of part 4.");
    return s.finish();
}
