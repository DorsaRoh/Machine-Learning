// P2 - indexing
//
// The most important exercise in the prerequisites. Everything in the main
// course -- coalescing, tiling, transposes, attention -- is this arithmetic
// with a thread index substituted for a loop variable.
//
// If you only do one of these six, do this one, and do it until it is boring.
//
// Chapter: P3

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "check.h"

// ================================================================ part 1
// Flattening and unflattening. One line each.

// Row-major: element (row, col) of a matrix with `num_cols` columns.
int flat_index(int row, int col, int num_cols) {
    // TODO
    (void)row;
    (void)col;
    (void)num_cols;
    return -1;
}

// The inverse: given a flat index, recover the row and the column.
int row_of(int flat, int num_cols) {
    // TODO
    (void)flat;
    (void)num_cols;
    return -1;
}

int col_of(int flat, int num_cols) {
    // TODO
    (void)flat;
    (void)num_cols;
    return -1;
}

// Column-major, which is what cuBLAS expects (Chapter 11).
int flat_index_colmajor(int row, int col, int num_rows) {
    // TODO
    (void)row;
    (void)col;
    (void)num_rows;
    return -1;
}

// ================================================================ part 2
// Reading rows and columns. Identical arithmetic, wildly different memory
// behaviour -- part 5 will show you how different.

// Sum of row `r` of an `rows x cols` row-major matrix.
double row_sum(const float* m, int rows, int cols, int r) {
    // TODO
    (void)m;
    (void)rows;
    (void)cols;
    (void)r;
    return 0.0;
}

// Sum of column `c`.
double col_sum(const float* m, int rows, int cols, int c) {
    // TODO
    (void)m;
    (void)rows;
    (void)cols;
    (void)c;
    return 0.0;
}

// ================================================================ part 3
// Transpose. out is cols x rows. Chapter 03 and Chapter 06 are both about
// making this fast; today just make it correct.
void transpose(const float* in, float* out, int rows, int cols) {
    // TODO
    (void)in;
    (void)out;
    (void)rows;
    (void)cols;
}

// ================================================================ part 4
// Strides, which are the general form of everything above.
//
// A tensor shaped [B, H, S, D] -- batch, heads, sequence, head-dim -- stored
// contiguously in that order. This is the exact layout of Q, K and V in
// Chapter 19.
int index_4d(int b, int h, int s, int d, int B, int H, int S, int D) {
    // TODO: note which of B, H, S, D you actually need. One of them is a
    // trap: the size of the outermost dimension never appears in the index.
    (void)b; (void)h; (void)s; (void)d;
    (void)B; (void)H; (void)S; (void)D;
    return -1;
}

// A view of a sub-rectangle. `m` is the full matrix with `ld` elements per row
// ("leading dimension"). The sub-matrix starts at (r0, c0). Return the flat
// index, into the FULL matrix, of element (i, j) of the sub-matrix.
int subview_index(int i, int j, int r0, int c0, int ld) {
    // TODO
    (void)i; (void)j; (void)r0; (void)c0; (void)ld;
    return -1;
}

// ================================================================ part 5
// Tiling with bounds clamping. A 16x16 tile at tile-coordinates (ti, tj) of a
// matrix whose size is NOT a multiple of 16.
//
// Fill `out` (16*16 floats, row-major) with the tile's contents, writing 0.0f
// for any position that falls outside the matrix.
//
// This is the `if (row < M && col < N)` guard that appears in every single
// kernel from Chapter 06 onward, and the reason it appears.
void load_tile(const float* m, int rows, int cols, int ti, int tj, float* out) {
    const int T = 16;
    // TODO
    (void)m; (void)rows; (void)cols; (void)ti; (void)tj; (void)out; (void)T;
}

// ================================================================ part 6
// Two loop orders over the same matrix. Same answer, different speed.
// Fill both in; the harness checks they agree and then times them.

void scale_row_major(float* m, int rows, int cols, float k) {
    // TODO: outer loop over rows, inner loop over columns.
    (void)m; (void)rows; (void)cols; (void)k;
}

void scale_col_major(float* m, int rows, int cols, float k) {
    // TODO: outer loop over columns, inner loop over rows.
    // Same multiplications, in a different order.
    (void)m; (void)rows; (void)cols; (void)k;
}

// ------------------------------------------------------------------------

template <typename F>
double time_ms(F f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    check::Suite s("p2-indexing");

    // ---- part 1 ----------------------------------------------------------
    s.eq_int("flat_index(0, 0, 8)", flat_index(0, 0, 8), 0);
    s.eq_int("flat_index(0, 5, 8)", flat_index(0, 5, 8), 5);
    s.eq_int("flat_index(1, 0, 8)", flat_index(1, 0, 8), 8);
    s.eq_int("flat_index(3, 2, 8)", flat_index(3, 2, 8), 26);
    s.eq_int("flat_index(3, 2, 100)", flat_index(3, 2, 100), 302);

    s.eq_int("row_of(26, 8)", row_of(26, 8), 3);
    s.eq_int("col_of(26, 8)", col_of(26, 8), 2);
    s.eq_int("row_of(7, 8)", row_of(7, 8), 0);
    s.eq_int("col_of(8, 8)", col_of(8, 8), 0);

    bool roundtrip = true;
    for (int r = 0; r < 7 && roundtrip; r++)
        for (int c = 0; c < 11 && roundtrip; c++) {
            int f = flat_index(r, c, 11);
            roundtrip = (row_of(f, 11) == r) && (col_of(f, 11) == c);
        }
    s.ok("flatten then unflatten is the identity, for all 77 cells", roundtrip);

    s.eq_int("colmajor(0, 1, 4) -- one column across", flat_index_colmajor(0, 1, 4), 4);
    s.eq_int("colmajor(1, 0, 4) -- one row down", flat_index_colmajor(1, 0, 4), 1);
    s.eq_int("colmajor(2, 3, 4)", flat_index_colmajor(2, 3, 4), 14);

    // ---- part 2 ----------------------------------------------------------
    // 4x5 matrix holding m[r][c] = 10*r + c.
    const int R = 4, C = 5;
    std::vector<float> m(R * C);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) m[r * C + c] = float(10 * r + c);

    s.eq_double("row_sum(row 0) = 0+1+2+3+4", row_sum(m.data(), R, C, 0), 10.0);
    s.eq_double("row_sum(row 2) = 20..24", row_sum(m.data(), R, C, 2), 110.0);
    s.eq_double("col_sum(col 0) = 0+10+20+30", col_sum(m.data(), R, C, 0), 60.0);
    s.eq_double("col_sum(col 4) = 4+14+24+34", col_sum(m.data(), R, C, 4), 76.0);

    double row_total = 0, col_total = 0;
    for (int r = 0; r < R; r++) row_total += row_sum(m.data(), R, C, r);
    for (int c = 0; c < C; c++) col_total += col_sum(m.data(), R, C, c);
    s.eq_double("all rows and all columns sum to the same total",
                row_total, col_total, 1e-9);

    // ---- part 3 ----------------------------------------------------------
    std::vector<float> t(C * R, -1.0f);
    transpose(m.data(), t.data(), R, C);
    std::vector<double> got(t.begin(), t.end()), want(C * R);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) want[c * R + r] = 10 * r + c;
    s.eq_matrix("transpose of a 4x5", got, want, C, R);

    // Transposing twice must return the original -- a cheap property test that
    // catches the "I transposed the indices but not the shape" bug.
    std::vector<float> back(R * C, -1.0f);
    transpose(t.data(), back.data(), C, R);
    std::vector<double> gb(back.begin(), back.end()), wb(m.begin(), m.end());
    s.eq_matrix("transpose twice is the identity", gb, wb, R, C);

    // ---- part 4 ----------------------------------------------------------
    // [B=2, H=3, S=5, D=4]
    s.eq_int("index_4d: first element", index_4d(0, 0, 0, 0, 2, 3, 5, 4), 0);
    s.eq_int("index_4d: next d is adjacent", index_4d(0, 0, 0, 1, 2, 3, 5, 4), 1);
    s.eq_int("index_4d: next s is D away", index_4d(0, 0, 1, 0, 2, 3, 5, 4), 4);
    s.eq_int("index_4d: next h is S*D away", index_4d(0, 1, 0, 0, 2, 3, 5, 4), 20);
    s.eq_int("index_4d: next b is H*S*D away", index_4d(1, 0, 0, 0, 2, 3, 5, 4), 60);
    s.eq_int("index_4d: last element", index_4d(1, 2, 4, 3, 2, 3, 5, 4), 119);

    s.eq_int("subview: origin maps to (r0, c0)", subview_index(0, 0, 2, 3, 16), 35);
    s.eq_int("subview: one row down", subview_index(1, 0, 2, 3, 16), 51);
    s.eq_int("subview: one column across", subview_index(0, 1, 2, 3, 16), 36);

    // ---- part 5 ----------------------------------------------------------
    // 20x20 matrix, so tile (1,1) is full but tile (1,1)'s neighbours are not.
    const int BR = 20, BC = 20;
    std::vector<float> big(BR * BC);
    for (int r = 0; r < BR; r++)
        for (int c = 0; c < BC; c++) big[r * BC + c] = float(r * 100 + c);

    std::vector<float> tile(256, -1.0f);
    load_tile(big.data(), BR, BC, 0, 0, tile.data());
    s.eq_double("tile(0,0): corner", tile[0], 0.0);
    s.eq_double("tile(0,0): element (3,5)", tile[3 * 16 + 5], 305.0);
    s.eq_double("tile(0,0): last element (15,15)", tile[15 * 16 + 15], 1515.0);

    std::fill(tile.begin(), tile.end(), -1.0f);
    load_tile(big.data(), BR, BC, 1, 1, tile.data());
    // Tile (1,1) covers rows 16..31 and cols 16..31; the matrix stops at 19.
    s.eq_double("tile(1,1): in-bounds corner is m[16][16]", tile[0], 1616.0);
    s.eq_double("tile(1,1): (3,3) is m[19][19], the last real cell",
                tile[3 * 16 + 3], 1919.0);
    s.eq_double("tile(1,1): (4,0) is off the bottom -> 0", tile[4 * 16 + 0], 0.0);
    s.eq_double("tile(1,1): (0,4) is off the right -> 0", tile[0 * 16 + 4], 0.0);
    s.eq_double("tile(1,1): (15,15) is far outside -> 0", tile[15 * 16 + 15], 0.0);

    // ---- part 6 ----------------------------------------------------------
    const int N = 4096;  // 64 MB
    std::vector<float> a(size_t(N) * N, 1.0f), b(size_t(N) * N, 1.0f);

    double t_row = time_ms([&] { scale_row_major(a.data(), N, N, 2.0f); });
    double t_col = time_ms([&] { scale_col_major(b.data(), N, N, 2.0f); });

    bool same = true;
    for (size_t i = 0; i < a.size() && same; i += 7919) same = (a[i] == b[i]);
    s.ok("both loop orders produce the same matrix", same);
    s.ok("both loop orders actually did the work (not left as TODO)",
         a[0] == 2.0f && b[0] == 2.0f);

    std::printf("\n  row-major order : %7.2f ms\n", t_row);
    std::printf("  col-major order : %7.2f ms\n", t_col);
    if (t_row > 0) std::printf("  ratio           : %7.2fx\n", t_col / t_row);
    std::printf("\n  Did you predict that number before running it?\n");

    s.note("The arithmetic in part 1 is the arithmetic in every kernel you will write.");
    return s.finish();
}
