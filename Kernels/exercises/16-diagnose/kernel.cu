// Exercise 16 - diagnose three slow kernels
//
// ====================  READ THIS BEFORE SCROLLING  ====================
//
// All three kernels below are CORRECT. All three are slow. Each is slow for a
// DIFFERENT reason.
//
// The exercise is to diagnose each one FROM PROFILER OUTPUT, before reading
// its source. So:
//
//   1. Run `make run` to see how slow each is.
//   2. Run `ncu --section SpeedOfLight --section WarpStateStats \
//          --section MemoryWorkloadAnalysis ./exercise`
//   3. Write down, for each kernel: which roof (if any) it is near, the top
//      stall reason, sectors per request, bank conflicts, occupancy.
//   4. From those numbers alone, write a one-sentence diagnosis.
//   5. ONLY THEN read the source and fix it.
//
// If you read the code first you will spot the bugs in thirty seconds, learn
// nothing, and still not be able to do this on a kernel you did not write.
//
// ======================================================================

#include <cuda_runtime.h>

// ================================================================== slow_1
// Computes the sum of each row of a [rows x cols] row-major matrix.
__global__ void slow_1(const float* __restrict__ A, float* __restrict__ out,
                       int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float s = 0.0f;
        for (int c = 0; c < cols; ++c) s += A[size_t(r) * cols + c];
        out[r] = s;
    }
}

// ================================================================== slow_2
// Transposes a square matrix through shared memory.
#define T2 32
__global__ void slow_2(const float* __restrict__ in, float* __restrict__ out,
                       int n) {
    __shared__ float tile[T2][T2];
    int x = blockIdx.x * T2 + threadIdx.x;
    int y = blockIdx.y * T2 + threadIdx.y;

    for (int j = 0; j < T2; j += blockDim.y)
        if (x < n && y + j < n) tile[threadIdx.y + j][threadIdx.x] = in[size_t(y + j) * n + x];
    __syncthreads();

    x = blockIdx.y * T2 + threadIdx.x;
    y = blockIdx.x * T2 + threadIdx.y;
    for (int j = 0; j < T2; j += blockDim.y)
        if (x < n && y + j < n) out[size_t(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ================================================================== slow_3
// Applies 8 rounds of an affine map to every element.
__global__ void slow_3(const float* __restrict__ x, float* __restrict__ y, int n) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? x[i] : 0.0f;
    for (int it = 0; it < 8; ++it) {
        s[threadIdx.x] = v;
        __syncthreads();
        v = s[threadIdx.x] * 1.0001f + 0.5f;
        __syncthreads();
    }
    if (i < n) y[i] = v;
}

// ---------------------------------------------------------------- your fixes
// Write the fixed versions here. Same signatures, same results, same meaning.
// Do not change what they compute.

__global__ void fast_1(const float* __restrict__ A, float* __restrict__ out,
                       int rows, int cols) {
    // TODO
}

__global__ void fast_2(const float* __restrict__ in, float* __restrict__ out,
                       int n) {
    // TODO
}

__global__ void fast_3(const float* __restrict__ x, float* __restrict__ y, int n) {
    // TODO
}

// ------------------------------------------------------------------ dispatch
void launch_1(bool fixed, const float* A, float* out, int rows, int cols) {
    int block = 256;
    if (!fixed) {
        slow_1<<<(rows + block - 1) / block, block>>>(A, out, rows, cols);
    } else {
        // TODO: your launch config for fast_1
    }
}

void launch_2(bool fixed, const float* in, float* out, int n) {
    dim3 b(32, 8), g((n + T2 - 1) / T2, (n + T2 - 1) / T2);
    if (!fixed) {
        slow_2<<<g, b>>>(in, out, n);
    } else {
        // TODO
    }
}

void launch_3(bool fixed, const float* x, float* y, int n) {
    int block = 256;
    if (!fixed) {
        slow_3<<<(n + block - 1) / block, block>>>(x, y, n);
    } else {
        // TODO
    }
}
