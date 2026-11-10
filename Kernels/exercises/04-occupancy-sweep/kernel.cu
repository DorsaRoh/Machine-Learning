// Exercise 04 - occupancy and ILP sweep
//
//   out[i] = in[i] * 2 + 1
//
// A deliberately trivial memory-bound body, swept across block size and ILP so
// that the only things changing are (a) how many warps are resident and (b)
// how many independent loads each thread has in flight.
//
// You write one kernel. It must process ILP elements per loop iteration with
// ILP INDEPENDENT loads -- that is the whole point. If the loads end up
// dependent on each other, or if the elements a thread handles are adjacent
// rather than grid-strided, the sweep measures something else.

#include <cuda_runtime.h>

template <int ILP>
__global__ void stream_kernel(const float* __restrict__ in,
                              float* __restrict__ out, size_t n) {
    // TODO: grid-stride loop, ILP elements per iteration.
    //
    // Two decisions that change the answer:
    //   1. Which ILP elements does one thread take in a single iteration?
    //      Adjacent (i, i+1, i+2, i+3) or spread by the thread count
    //      (i, i+T, i+2T, i+3T)? One of these keeps the warp coalesced.
    //      Work out which before you type.
    //   2. Issue all ILP loads into separate variables BEFORE using any of
    //      them. If you load-use-load-use, you have ILP=1 wearing a costume.
    //
    // Do not forget the tail: n is not a multiple of ILP * grid * block.
}

// Dispatch helper (given). Turns a runtime ILP into a compile-time template
// argument, because the unrolling has to happen at compile time to be real.
inline void launch_stream(const float* in, float* out, size_t n, int block,
                          int grid, int ilp) {
    switch (ilp) {
        case 1: stream_kernel<1><<<grid, block>>>(in, out, n); break;
        case 2: stream_kernel<2><<<grid, block>>>(in, out, n); break;
        case 4: stream_kernel<4><<<grid, block>>>(in, out, n); break;
        case 8: stream_kernel<8><<<grid, block>>>(in, out, n); break;
        default: stream_kernel<1><<<grid, block>>>(in, out, n); break;
    }
}
