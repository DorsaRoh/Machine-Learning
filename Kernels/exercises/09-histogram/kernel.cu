// Exercise 09 - histogram under contention
//
//   hist[in[i]]++   for i in [0, n),  in[i] in [0, nbins)
//
// The harness runs your kernel on three input distributions: uniform, moderately
// skewed, and brutally skewed (90% of elements in one bin). Same kernel, same
// size. The spread between those three numbers is what this exercise is about.
//
// A naive global-atomic version will pass correctness and fail the bar on at
// least one distribution. Your job is a kernel whose performance does not
// collapse when the data is ugly.
//
// nbins is at most 1024 here, so a private copy fits in shared memory. Decide
// whether one copy per block is enough.

#include <cuda_runtime.h>

__global__ void histogram(const unsigned int* __restrict__ in,
                          unsigned int* __restrict__ hist, int n, int nbins) {
    // TODO.
    //
    // Checklist to work through before writing:
    //   - private or global? (Chapter 09 section 3)
    //   - how do you zero the private copy, and what must follow that?
    //   - are your reads of `in` coalesced?
    //   - how many elements per thread?
    //   - warp aggregation: __match_any_sync. Worth it here? At which skew?
    //   - replication: R copies in shared memory, thread t uses copy t % R.
    //     What does R cost you, and what does it buy?
    //   - the merge back to global: how many atomics total?
}

void launch_histogram(const unsigned int* in, unsigned int* hist, int n,
                      int nbins) {
    int block = 256;
    int grid = 1;  // TODO
    histogram<<<grid, block>>>(in, hist, n, nbins);
}
