// Exercise 07 - the reduction ladder
//
// Sum n fp32 values. Seven versions, each fixing one specific inefficiency in
// the previous one. The harness runs whichever versions you have implemented
// and prints the speedup chain.
//
// Chapter 07 section 3 tells you WHAT each step addresses, as a question. It
// does not tell you the code. Answer the question, then write it.
//
// Rules:
//   - Every version must produce the sum in out[0].
//   - `scratch` is a device buffer of at least `max_blocks` floats, for
//     per-block partials if your version wants them.
//   - You may launch more than one kernel per version. The harness times the
//     whole launch function, which is the honest thing to measure.
//
// Set IMPLEMENTED to the highest version you have written.

#include <cuda_runtime.h>

#define IMPLEMENTED 1  // TODO: raise this as you climb

// -------------------------------------------------------------------------
// v1: interleaved addressing with modulo.
//     Tree in shared memory, `tid % (2*s) == 0` selects active threads.
// -------------------------------------------------------------------------
__global__ void reduce_v1(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// v2: interleaved without modulo. Same tree, contiguous active threads.
// -------------------------------------------------------------------------
__global__ void reduce_v2(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// v3: sequential addressing.  `tid < s`, s halving from blockDim/2.
// -------------------------------------------------------------------------
__global__ void reduce_v3(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// v4: first add during load. Each thread loads 2 elements.
// -------------------------------------------------------------------------
__global__ void reduce_v4(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// v5: shuffle the last warp. Below 32 active threads, leave shared memory.
//     NOTE: the pre-Volta "volatile shared, no barrier" trick is ILLEGAL now
//     (Chapter 05). Use __shfl_down_sync.
// -------------------------------------------------------------------------
__global__ void reduce_v5(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// v6: full unroll. Block size as a template parameter so every bound is a
//     compile-time constant. (Change the signature to a template if you want;
//     adjust the dispatch below.)
// -------------------------------------------------------------------------
__global__ void reduce_v6(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// v7: grid-stride. A fixed grid sized to the GPU; each thread sums MANY
//     elements before the tree runs at all.
//     This is the one that reaches bandwidth. Before writing it, work out the
//     ratio of memory work to tree work in v6 versus v7.
// -------------------------------------------------------------------------
__global__ void reduce_v7(const float* __restrict__ in, float* __restrict__ out,
                          size_t n) {
    // TODO
}

// -------------------------------------------------------------------------
// Dispatch. `out` is a single float on the device. `scratch` has room for
// `max_blocks` floats. You decide grid/block per version, and whether to run
// a second pass over the partials.
// -------------------------------------------------------------------------
void launch_reduce(int version, const float* in, float* out, float* scratch,
                   size_t n, int max_blocks) {
    // TODO: pick grid/block per version, run the kernel(s), leave the total
    // in out[0].
    //
    // A note on honesty: if your version writes per-block partials to
    // `scratch` and then reduces them with a second launch, BOTH launches are
    // inside the timed region. That is correct -- the extra pass is a real
    // cost of that design, and seeing it is part of the exercise.
    (void)version; (void)in; (void)out; (void)scratch; (void)n; (void)max_blocks;
}
