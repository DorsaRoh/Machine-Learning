// Exercise 02 - SAXPY
//
//   y[i] = alpha * x[i] + y[i]
//
// Same shape as vector add, one critical difference: y is both read and
// written. Before you write a line, work out whether this moves more, less, or
// the same traffic as vector add, and whether it should therefore be faster,
// slower, or identical.
//
// Then make it hit the bar.

#include <cuda_runtime.h>

__global__ void saxpy(float alpha, const float* __restrict__ x,
                      float* __restrict__ y, int n) {
    // TODO
}

void launch_saxpy(float alpha, const float* x, float* y, int n) {
    int block = 256;
    int grid = 1;  // TODO
    saxpy<<<grid, block>>>(alpha, x, y, n);
}
