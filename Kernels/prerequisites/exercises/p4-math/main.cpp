// P4 - math
//
// Every operation you will spend the next six months making fast, written the
// slow obvious way on a CPU. By the end of this file you will have implemented
// attention. It will take about forty lines and run at roughly one-millionth
// of the speed a GPU manages, and that gap is the whole course.
//
// Chapter: P7

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"

// ================================================================ part 1
// Reductions: many numbers in, one number out.

double dot(const float* a, const float* b, int n) {
    // TODO. Accumulate in double.
    (void)a; (void)b; (void)n;
    return 0.0;
}

// Index of the largest element. On a tie, the smallest index wins --
// that tie rule is not pedantry, it is what makes a parallel argmax
// reproducible (Chapter 07).
int argmax(const float* a, int n) {
    // TODO
    (void)a; (void)n;
    return -1;
}

// L2 norm: sqrt of the sum of squares.
double l2_norm(const float* a, int n) {
    // TODO
    (void)a; (void)n;
    return 0.0;
}

// ================================================================ part 2
// Matrix multiply. C is M x N, A is M x K, B is K x N, all row-major.
// C = A @ B.  Three loops. No cleverness yet.
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}

// C = A @ B^T.  A is M x K, B is N x K (note: N rows, K columns), C is M x N.
// You must NOT build the transpose -- just index B differently.
//
// This is the shape of Q @ K^T in attention, and the reason a GEMM kernel
// takes "transpose A?" and "transpose B?" flags instead of transposing.
void matmul_bt(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
}

// How many floating-point operations does an M x N x K matmul perform?
// Counting a multiply and an add as two. You will use this constantly.
long long matmul_flops(int M, int N, int K) {
    // TODO
    (void)M; (void)N; (void)K;
    return 0;
}

// How many bytes must it read and write, at minimum, if everything is fp32
// and each element is touched once?
long long matmul_bytes(int M, int N, int K) {
    // TODO
    (void)M; (void)N; (void)K;
    return 0;
}

// ================================================================ part 3
// Row-wise operations on an R x C matrix. Each row is independent, which is
// exactly why one GPU block per row works so well (Chapter 10).

// Numerically stable softmax, applied to each row independently.
void softmax_rows(const float* in, float* out, int R, int C) {
    // TODO
    (void)in; (void)out; (void)R; (void)C;
}

// LayerNorm: for each row, subtract the mean, divide by sqrt(variance + eps),
// then scale by gamma and shift by beta (both length C).
void layernorm_rows(const float* in, const float* gamma, const float* beta,
                    float* out, int R, int C, float eps) {
    // TODO
    (void)in; (void)gamma; (void)beta; (void)out; (void)R; (void)C; (void)eps;
}

// RMSNorm: no mean subtraction at all. Divide by sqrt(mean(x^2) + eps), then
// scale by gamma. Used by Llama and friends because it is one pass cheaper.
void rmsnorm_rows(const float* in, const float* gamma, float* out, int R, int C,
                  float eps) {
    // TODO
    (void)in; (void)gamma; (void)out; (void)R; (void)C; (void)eps;
}

// ================================================================ part 4
// Attention. Q, K, V are each S x D, row-major. O is S x D.
//
//     O = softmax( (Q @ K^T) * scale ) @ V      with scale = 1/sqrt(D)
//
// If `causal` is true, position i may only attend to positions j <= i.
//
// You may allocate scratch space for the S x S score matrix. That allocation
// is precisely what FlashAttention exists to eliminate -- at S = 8192 the
// scores alone are 256 MB per head, which is why Chapter 19 is a whole
// chapter rather than a paragraph.
void attention(const float* Q, const float* K, const float* V, float* O, int S,
               int D, bool causal) {
    // TODO
    (void)Q; (void)K; (void)V; (void)O; (void)S; (void)D; (void)causal;
}

// ------------------------------------------------------------------------

static std::vector<double> as_double(const std::vector<float>& v) {
    return std::vector<double>(v.begin(), v.end());
}

int main() {
    check::Suite s("p4-math");

    // ---- part 1 ----------------------------------------------------------
    std::vector<float> a = {1, 2, 3, 4, 5}, b = {2, 2, 2, 2, 2};
    s.eq_double("dot({1..5}, {2,2,2,2,2})", dot(a.data(), b.data(), 5), 30.0);
    s.eq_double("dot of an empty range", dot(a.data(), b.data(), 0), 0.0);

    // 1000 terms, each exactly 1. A float accumulator would survive this one;
    // the next one is the test.
    std::vector<float> inv(1000), ramp(1000);
    for (int i = 0; i < 1000; i++) {
        inv[i] = 1.0f / float(i + 1);
        ramp[i] = float(i + 1);
    }
    // The true answer is exactly 1000, but 1/(i+1) is not exactly
    // representable, so even a perfect implementation lands a few parts per
    // billion away. Hence the tolerance. This is P4 showing up uninvited.
    s.eq_double("dot(1/i, i) over 1000 terms", dot(inv.data(), ramp.data(), 1000),
                1000.0, 1e-3);

    std::vector<float> peaky = {1.0f, -3.0f, 7.0f, 2.0f, 0.0f};
    s.eq_int("argmax picks the max", argmax(peaky.data(), 5), 2);
    std::vector<float> tied = {1.0f, 7.0f, 3.0f, 7.0f, 2.0f};
    s.eq_int("argmax breaks ties toward the lower index",
             argmax(tied.data(), 5), 1);
    std::vector<float> neg = {-5.0f, -1.0f, -9.0f};
    s.eq_int("argmax on all-negative data", argmax(neg.data(), 3), 1);

    std::vector<float> three_four = {3.0f, 4.0f};
    s.eq_double("l2_norm({3,4})", l2_norm(three_four.data(), 2), 5.0, 1e-6);

    // ---- part 2 ----------------------------------------------------------
    // A is 2x3 holding 0..5, B is 3x4 holding 0..11.
    std::vector<float> A = {0, 1, 2, 3, 4, 5};
    std::vector<float> B = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<float> C(2 * 4, -1.0f);
    matmul(A.data(), B.data(), C.data(), 2, 4, 3);
    s.eq_matrix("matmul 2x3 @ 3x4", as_double(C),
                {20, 23, 26, 29, 56, 68, 80, 92}, 2, 4);

    // A @ I must be A.
    const int n4 = 4;
    std::vector<float> M4(n4 * n4), I4(n4 * n4, 0.0f), P4(n4 * n4, -1.0f);
    for (int i = 0; i < n4 * n4; i++) M4[i] = float(i * 3 % 7) - 2.0f;
    for (int i = 0; i < n4; i++) I4[i * n4 + i] = 1.0f;
    matmul(M4.data(), I4.data(), P4.data(), n4, n4, n4);
    s.eq_matrix("A @ I == A", as_double(P4), as_double(M4), n4, n4);

    // I @ A must also be A. This one catches a swapped index pair that the
    // previous check cannot.
    std::fill(P4.begin(), P4.end(), -1.0f);
    matmul(I4.data(), M4.data(), P4.data(), n4, n4, n4);
    s.eq_matrix("I @ A == A", as_double(P4), as_double(M4), n4, n4);

    // Bt is B transposed: 4 rows of 3.
    std::vector<float> Bt(4 * 3);
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 4; j++) Bt[j * 3 + k] = B[k * 4 + j];
    std::vector<float> C2(2 * 4, -1.0f);
    matmul_bt(A.data(), Bt.data(), C2.data(), 2, 4, 3);
    s.eq_matrix("matmul_bt agrees with matmul", as_double(C2),
                {20, 23, 26, 29, 56, 68, 80, 92}, 2, 4);

    s.eq_int("matmul_flops(1024,1024,1024)", matmul_flops(1024, 1024, 1024),
             2147483648LL);
    s.eq_int("matmul_flops(4096,4096,4096) -- note it exceeds an int",
             matmul_flops(4096, 4096, 4096), 137438953472LL);
    s.eq_int("matmul_bytes(1024,1024,1024)", matmul_bytes(1024, 1024, 1024),
             12582912LL);

    // ---- part 3 ----------------------------------------------------------
    // Row 1 is shifted by 1000 to catch an unstable softmax.
    std::vector<float> sm_in = {1, 2, 3, 1000, 1000, 1001};
    std::vector<float> sm_out(6, -1.0f);
    softmax_rows(sm_in.data(), sm_out.data(), 2, 3);
    s.eq_matrix("softmax_rows", as_double(sm_out),
                {0.09003057317, 0.2447284711, 0.6652409558, 0.2119415576,
                 0.2119415576, 0.5761168848},
                2, 3, 1e-6);
    s.eq_double("row 0 sums to 1", sm_out[0] + sm_out[1] + sm_out[2], 1.0, 1e-6);
    s.eq_double("row 1 sums to 1", sm_out[3] + sm_out[4] + sm_out[5], 1.0, 1e-6);

    std::vector<float> ln_in = {1, 2, 3, 4};
    std::vector<float> ones4 = {1, 1, 1, 1}, zeros4 = {0, 0, 0, 0};
    std::vector<float> ln_out(4, -1.0f);
    layernorm_rows(ln_in.data(), ones4.data(), zeros4.data(), ln_out.data(), 1, 4,
                   1e-5f);
    s.eq_matrix("layernorm, gamma=1 beta=0", as_double(ln_out),
                {-1.34163542, -0.4472118067, 0.4472118067, 1.34163542}, 1, 4,
                1e-5);

    std::vector<float> gamma = {2.0f, 0.5f, 1.0f, -1.0f};
    std::vector<float> beta = {0.0f, 1.0f, -1.0f, 0.5f};
    layernorm_rows(ln_in.data(), gamma.data(), beta.data(), ln_out.data(), 1, 4,
                   1e-5f);
    s.eq_matrix("layernorm with gamma and beta", as_double(ln_out),
                {-2.68327084, 0.7763940967, -0.5527881933, -0.84163542}, 1, 4,
                1e-5);

    // A constant row: variance 0. Without eps this divides by zero.
    std::vector<float> flat_row = {5, 5, 5, 5};
    layernorm_rows(flat_row.data(), ones4.data(), zeros4.data(), ln_out.data(), 1,
                   4, 1e-5f);
    s.eq_matrix("layernorm of a constant row is all zeros, not NaN",
                as_double(ln_out), {0, 0, 0, 0}, 1, 4, 1e-4);

    std::vector<float> rms_out(4, -1.0f);
    rmsnorm_rows(ln_in.data(), ones4.data(), rms_out.data(), 1, 4, 1e-5f);
    s.eq_matrix("rmsnorm", as_double(rms_out),
                {0.3651481282, 0.7302962565, 1.095444385, 1.460592513}, 1, 4,
                1e-5);

    // ---- part 4 ----------------------------------------------------------
    const int S = 3, D = 2;
    std::vector<float> Q = {1, 0, 0, 1, 1, 1};
    std::vector<float> Kk = {1, 0, 0, 1, -1, 2};
    std::vector<float> Vv = {1, 2, 3, 4, 5, 6};
    std::vector<float> O(S * D, -1.0f);

    attention(Q.data(), Kk.data(), Vv.data(), O.data(), S, D, false);
    s.eq_matrix("attention, non-causal", as_double(O),
                {2.1281078, 3.1281078, 3.8718922, 4.8718922, 3.0, 4.0}, S, D,
                1e-5);

    std::fill(O.begin(), O.end(), -1.0f);
    attention(Q.data(), Kk.data(), Vv.data(), O.data(), S, D, true);
    s.eq_matrix("attention, causal", as_double(O),
                {1.0, 2.0, 2.339523099, 3.339523099, 3.0, 4.0}, S, D, 1e-5);

    // Query 2 attends equally to all three keys, so its output must be the
    // plain average of V. If you forgot the 1/sqrt(D) scale, this row still
    // passes and the first two do not -- which is how you will know.
    s.eq_double("row 2 is the mean of V, dim 0", O[4], 3.0, 1e-5);

    std::printf("\n  You just wrote attention. Chapter 19 makes it 10x faster\n");
    std::printf("  by never storing that S x S score matrix at all.\n");

    s.note("Everything here is correct and slow. The rest of the course is the other half.");
    return s.finish();
}
