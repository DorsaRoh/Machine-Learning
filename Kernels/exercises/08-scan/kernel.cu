// Exercise 08 - exclusive prefix sum
//
//   out[i] = in[0] + in[1] + ... + in[i-1],  out[0] = 0
//
// The bar is 60% of your copy ceiling, which is deliberately set so that a
// textbook three-kernel scan CANNOT pass. Count the passes before you choose
// an algorithm.
//
// `scratch` is a device buffer of at least `scratch_floats` floats, for block
// sums, flags, status words, or whatever your design needs.
//
// You may use any algorithm. Suggested order of attack:
//   1. warp scan with __shfl_up_sync
//   2. block scan built from warp scans
//   3. the three-kernel decomposition (correct, and too slow -- measure it)
//   4. single pass: decoupled look-back  (Chapter 08 section 5)
//
// Step 4 needs real memory-ordering care. If you take it on, Chapter 18 is
// worth reading early, and `compute-sanitizer --tool racecheck` is your
// friend.

#include <cuda_runtime.h>

// TODO: your kernels here.

void launch_scan(const float* in, float* out, void* scratch,
                 size_t scratch_bytes, size_t n) {
    // TODO: leave the exclusive scan of `in` in `out`.
    (void)in; (void)out; (void)scratch; (void)scratch_bytes; (void)n;
}

// Called once before timing, so a look-back implementation can zero its status
// words outside the measured region (which is what a real library does too).
void reset_scan_state(void* scratch, size_t scratch_bytes) {
    (void)scratch; (void)scratch_bytes;
}
