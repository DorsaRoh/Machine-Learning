// Exercise 24 - one kernel, shipped.
//
// The op:
//
//     out[r, :] = x[r, :] + gelu( layernorm(x[r, :]) * w + b )
//
// which is the memory-bound half of a transformer block. Eager PyTorch runs it
// as three kernels and moves the tensor about seven times; you are going to
// move it twice.
//
// This file is compiled by ext.py via torch.utils.cpp_extension.load.
//
// Use tanh-approximate GELU, which is what PyTorch's F.gelu(approximate='tanh')
// computes, and what the harness compares against:
//
//     gelu(v) = 0.5 * v * (1 + tanh( sqrt(2/pi) * (v + 0.044715 v^3) ))

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

__global__ void fused_ln_gelu_res_kernel(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         const float* __restrict__ b,
                                         float* __restrict__ out, int rows,
                                         int cols, float eps) {
    // TODO.
    //
    // One block per row is the obvious decomposition. Before you write it:
    //   - how many times must this kernel read x? (chapter 10 section 3)
    //   - Welford, two-pass, or sum/sumsq? (chapter 15 section 2 -- and the
    //     harness tests a shifted distribution on purpose)
    //   - where do w and b live? they are read by every row
    //   - is cols small enough to hold a row in registers? what if it is not?
}

torch::Tensor fused_ln_gelu_res(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                                double eps) {
    // TODO: the checks that stop this from silently producing garbage when
    // PyTorch hands you something you did not expect. At minimum: device,
    // dtype, contiguity, and that w and b match x's last dimension.
    //
    // Then allocate with torch::empty_like (the caching allocator, not
    // cudaMalloc) and launch on c10::cuda::getCurrentCUDAStream().

    auto out = torch::empty_like(x);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_gelu_res", &fused_ln_gelu_res,
          "fused layernorm + gelu + residual");
}
