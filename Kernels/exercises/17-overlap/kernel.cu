// Exercise 17 - the copy-compute pipeline
//
// Process a buffer larger than you want to hold on the GPU at once, in chunks:
//
//     for each chunk:  H2D copy  ->  kernel  ->  D2H copy
//
// The serial version does those strictly in order. Your job is the pipelined
// version, where chunk k's compute overlaps chunk k+1's copy.
//
// Gate: 1.6x over the serial version.
//
// The kernel itself is deliberately boring -- a few FMAs per element -- so that
// the copy and the compute take comparable time and overlapping actually pays.
// Do not optimise the kernel; optimise the schedule.

#include <cuda_runtime.h>

__global__ void process(const float* __restrict__ in, float* __restrict__ out,
                        int n, int work) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        for (int k = 0; k < work; ++k) v = v * 1.0001f + 0.0001f;
        out[i] = v;
    }
}

// Both versions must produce identical results in h_out.
//
// h_in and h_out are host buffers of `total` floats.
// d_buf is a device scratch buffer of `n_streams * chunk * 2` floats:
//   use the first half for inputs and the second for outputs, or carve it up
//   however your design wants.

void run_serial(const float* h_in, float* h_out, float* d_buf, size_t total,
                size_t chunk, int work) {
    // TODO: the obvious version. cudaMemcpy, kernel, cudaMemcpy, repeat.
    // This is the baseline the gate compares against, so write it honestly.
    (void)h_in; (void)h_out; (void)d_buf; (void)total; (void)chunk; (void)work;
}

void run_pipelined(const float* h_in, float* h_out, float* d_buf, size_t total,
                   size_t chunk, int work, cudaStream_t* streams,
                   int n_streams) {
    // TODO.
    //
    // Requirements for this to actually overlap, all of which are easy to get
    // wrong and silently lose:
    //   - the HOST buffers must be pinned (the harness pins them for you)
    //   - cudaMemcpyAsync, not cudaMemcpy
    //   - every operation on an explicit stream, never the default stream
    //   - each stream needs its OWN device staging buffers, or chunk k+1 will
    //     overwrite chunk k's data while it is still being used
    //   - synchronise once at the end, not per chunk
    //
    // Check the result in nsys, not just the number: you should SEE three rows
    // of activity overlapping.
    (void)h_in; (void)h_out; (void)d_buf; (void)total; (void)chunk;
    (void)work; (void)streams; (void)n_streams;
}
