// Exercise 06 - transpose, three ways
//
// Same output, three implementations, measured side by side:
//
//   v1  naive          global read coalesced, global write strided
//   v2  tiled          both global accesses coalesced, shared column access
//                      32-way bank conflicted
//   v3  tiled + padded both global accesses coalesced, no bank conflicts
//
// The whole point is the DIFFERENCES. Implement all three even though you know
// v3 wins, because the gaps v1->v2 and v2->v3 are the two lessons of
// Chapter 06, and you only learn their size by measuring them.

#include <cuda_runtime.h>

#define TILE 32
#define BLOCK_ROWS 8  // 32x8 threads; each thread moves TILE/BLOCK_ROWS = 4 elements

// ---------------------------------------------------------------- v1 naive
__global__ void transpose_v1(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
    // TODO: the Chapter 03 version. You may paste your own from
    // 03-transpose-naive -- it is here as the baseline to beat.
}

// ---------------------------------------------------------------- v2 tiled
__global__ void transpose_v2(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
    __shared__ float tile[TILE][TILE];

    // TODO:
    //  1. compute the input tile origin from blockIdx
    //  2. load TILE x TILE into `tile`, coalesced, looping BLOCK_ROWS at a
    //     time over the tile's rows
    //  3. __syncthreads()
    //  4. compute the OUTPUT tile origin -- it is not the same as the input
    //     origin, and getting this wrong is the single most common bug here.
    //     Which block's output tile are you writing?
    //  5. write out, coalesced, reading `tile` transposed
    //
    // Before writing step 5: which of your two shared accesses hits bank
    // conflicts, and how many ways? Predict it, then confirm with ncu.
}

// -------------------------------------------------------- v3 tiled + padded
__global__ void transpose_v3(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
    __shared__ float tile[TILE][TILE + 1];  // the only change from v2

    // TODO: identical body to v2.
    //
    // Work out on paper why one extra column removes the conflict before you
    // measure it. "It just does" is not an answer you can carry to the swizzle
    // in Chapter 13.
}

void launch_transpose(int version, const float* in, float* out, int n) {
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    switch (version) {
        case 1: {
            dim3 block(TILE, BLOCK_ROWS);
            transpose_v1<<<grid, block>>>(in, out, n);
            break;
        }
        case 2: {
            dim3 block(TILE, BLOCK_ROWS);
            transpose_v2<<<grid, block>>>(in, out, n);
            break;
        }
        case 3: {
            dim3 block(TILE, BLOCK_ROWS);
            transpose_v3<<<grid, block>>>(in, out, n);
            break;
        }
    }
}
