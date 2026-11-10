// Exercise 18 - race hunt
//
// Four kernels. Each has exactly one synchronisation bug. Some of them will
// give the right answer on your first run, which is the entire point.
//
// For each: find the bug, say when it manifests, say which compute-sanitizer
// tool finds it (one of them is found by no tool), and fix it WITHOUT changing
// what the kernel computes.
//
// The harness runs each many times and reports how many runs disagreed with
// the reference and whether the results were bit-identical across runs.
//
// Run `make sanitize` early and often.

#include <cuda_runtime.h>

#define N_PER_BLOCK 256

// ============================================================== bug_1
// Sliding-window sum: out[i] = in[i] + in[i+1] + in[i+2] + in[i+3].
// Each block walks several tiles, reusing one shared buffer.
__global__ void bug_1(const float* __restrict__ in, float* __restrict__ out,
                      int n, int tiles) {
    __shared__ float s[N_PER_BLOCK + 4];
    int base = blockIdx.x * N_PER_BLOCK;

    for (int t = 0; t < tiles; ++t) {
        int g = base + t * gridDim.x * N_PER_BLOCK + threadIdx.x;
        s[threadIdx.x] = (g < n) ? in[g] : 0.0f;
        if (threadIdx.x < 4)
            s[N_PER_BLOCK + threadIdx.x] =
                (g + N_PER_BLOCK < n) ? in[g + N_PER_BLOCK] : 0.0f;
        __syncthreads();

        float acc = s[threadIdx.x] + s[threadIdx.x + 1] + s[threadIdx.x + 2] +
                    s[threadIdx.x + 3];
        if (g < n) out[g] = acc;
    }
}

// ============================================================== bug_2
// Each block reduces its own slice and adds the result into a per-group
// accumulator. Many blocks share each accumulator.
//
// The values are small integers so that float addition is exact: a correct
// version is bit-reproducible no matter what order the blocks finish in.
__global__ void bug_2(float* __restrict__ out, int groups) {
    __shared__ float s[N_PER_BLOCK];
    int i = blockIdx.x * N_PER_BLOCK + threadIdx.x;
    s[threadIdx.x] = float((i * 37) % 11);
    __syncthreads();

    for (int stride = N_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int g = blockIdx.x % groups;
        out[g] += s[0];
    }
}

// ============================================================== bug_3
// Block reduction with the classic "unroll the last warp" trick, copied
// verbatim from a tutorial written before 2017. It was correct then.
__global__ void bug_3(const float* __restrict__ in, float* __restrict__ out,
                      int n) {
    __shared__ volatile float s[N_PER_BLOCK];
    int i = blockIdx.x * N_PER_BLOCK + threadIdx.x;
    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = N_PER_BLOCK / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        s[threadIdx.x] += s[threadIdx.x + 32];
        s[threadIdx.x] += s[threadIdx.x + 16];
        s[threadIdx.x] += s[threadIdx.x + 8];
        s[threadIdx.x] += s[threadIdx.x + 4];
        s[threadIdx.x] += s[threadIdx.x + 2];
        s[threadIdx.x] += s[threadIdx.x + 1];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

// ============================================================== bug_4
// Block 0 computes a scale factor and publishes it; every other block waits
// for it and uses it. Someone was careful enough to reach for `volatile`.
__global__ void bug_4(const float* __restrict__ in, float* __restrict__ out,
                      float* __restrict__ scale, int* __restrict__ ready, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            *scale = 2.0f;
            *ready = 1;
        }
    } else {
        if (threadIdx.x == 0)
            while (*((volatile int*)ready) == 0) {
            }
        __syncthreads();
    }
    if (i < n) out[i] = in[i] * (*scale);
}

// ------------------------------------------------------------- your fixes
// Same signatures, same computed result, no races.

__global__ void fix_1(const float* __restrict__ in, float* __restrict__ out,
                      int n, int tiles) {
    // TODO
}

__global__ void fix_2(float* __restrict__ out, int groups) {
    // TODO
}

__global__ void fix_3(const float* __restrict__ in, float* __restrict__ out,
                      int n) {
    // TODO
}

__global__ void fix_4(const float* __restrict__ in, float* __restrict__ out,
                      float* __restrict__ scale, int* __restrict__ ready, int n) {
    // TODO
}
