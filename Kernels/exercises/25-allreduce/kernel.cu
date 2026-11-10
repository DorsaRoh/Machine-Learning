// Exercise 25 - ring AllReduce by hand
//
// Every GPU starts with its own buffer of n floats. Every GPU must end with
// the element-wise sum of all of them.
//
// The ring (chapter 25 section 2):
//
//   split the buffer into ngpu chunks
//   phase 1, reduce-scatter : ngpu-1 steps. In step s, GPU g sends chunk
//       (g - s) mod ngpu to GPU g+1, which adds it into its own copy.
//       After ngpu-1 steps, GPU g owns the fully reduced chunk (g+1) mod ngpu.
//   phase 2, all-gather     : ngpu-1 steps circulating the reduced chunks.
//
//   bytes per GPU = 2 * (ngpu-1)/ngpu * n * 4
//
// Compute that number, divide by your measured P2P bandwidth, and you have
// your target before you write a line.
//
// Peer access is already enabled by the harness wherever the topology allows
// it, so cudaMemcpyPeerAsync will take the direct path when there is one.
//
// Gate: exact, and >= 60% of NCCL's bandwidth at 64 MB.

#include <cuda_runtime.h>

__global__ void add_into(float* __restrict__ dst, const float* __restrict__ src,
                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// bufs[g]    : n floats on device g. Holds the input on entry, must hold the
//              sum across all GPUs on exit.
// scratch[g] : n floats on device g, yours to use.
// streams[g] : a stream on device g. Everything must go on these; the harness
//              synchronizes them when it wants a result.
void ring_allreduce(float** bufs, float** scratch, int n, int ngpu,
                    cudaStream_t* streams) {
    // TODO.
    //
    // Before writing, answer these on paper -- a ring is easy to write and
    // very easy to write subtly wrong:
    //   - in step s, which chunk does GPU g send, and where does it land?
    //   - what stops GPU g+1 from adding a chunk before it has fully arrived?
    //   - what stops GPU g from overwriting a chunk GPU g-1 is still reading?
    //   - how many synchronizations per step do you actually need, and does
    //     cudaStreamSynchronize or cudaEventRecord/cudaStreamWaitEvent cost
    //     less here?
    //   - cudaSetDevice before each operation: which operations care?
    //
    // The naive version synchronizes every device after every step. Write
    // that, verify it, measure it, and then remove synchronizations one at a
    // time, re-verifying after each. Do not start with the clever version.
    (void)bufs; (void)scratch; (void)n; (void)ngpu; (void)streams;
}
