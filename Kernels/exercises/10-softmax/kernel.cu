// Exercise 10 - row-wise softmax
//
//   y[r][c] = exp(x[r][c] - m_r) / sum_c exp(x[r][c] - m_r),  m_r = max_c x[r][c]
//
// Memory-bound. The bar is 85% of copy ceiling over 2 * rows * cols * 4 bytes,
// which means ONE read and ONE write. Any implementation that reads the row
// three times is capped at ~33% and cannot pass.
//
// The harness tests three shapes:
//   [8192, 1024]    the typical transformer case
//   [262144, 32]    many tiny rows
//   [64, 131072]    few enormous rows  (a row does not fit on-chip)
//
// One decomposition will not be best for all three. You may branch on the
// shape inside launch_softmax -- real libraries do exactly that.

#include <cuda_runtime.h>

// TODO: your kernel(s).
//
// Design questions to answer on paper first (Chapter 10 section 4):
//   - block per row, warp per row, or multiple blocks per row? For which shape?
//   - how many passes over the row does your design make?
//   - if you keep the row in registers between passes, how many registers is
//     that, and what does it do to occupancy at cols = 1024?
//   - for cols = 131072 the row does not fit anywhere on-chip. What then?
//     (This is where the online update from Chapter 10 section 2 earns itself.)
//   - __expf vs expf: what is the accuracy difference, and does it matter
//     against a 1e-5 tolerance?

void launch_softmax(const float* x, float* y, int rows, int cols) {
    // TODO
    (void)x; (void)y; (void)rows; (void)cols;
}
