"""Exercise 23 - the Triton ladder.

Four kernels you have already written in CUDA. Write them again, in Triton,
and find out what the compiler does for you and what it does not.

Rules:
  - every kernel must handle n that is not a multiple of BLOCK
  - no torch ops inside the kernels; tl.* only
  - the launcher wrappers return a new tensor, do not modify inputs

Chapter: ../../chapters/23-triton.md
"""

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------- 1. add
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # TODO
    pass


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    # TODO: grid, BLOCK, launch
    return out


# ------------------------------------------------------------- 2. reduce
@triton.jit
def sum_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr):
    # TODO. Note what you do NOT have to write here compared with 07-reduce:
    # no shared memory, no __syncthreads, no shuffle ladder. tl.sum is the
    # whole reduction. Ask yourself what it compiled to, and check.
    pass


def total_sum(x: torch.Tensor) -> torch.Tensor:
    # TODO. One kernel or two? The same decomposition question as 07-reduce.
    raise NotImplementedError


# ------------------------------------------------------------ 3. softmax
@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK: tl.constexpr):
    # TODO. One program per row.
    #
    # The easy version loads the whole row at once, which caps n_cols at
    # whatever BLOCK your registers allow. The harness tests a shape where
    # that fails. Decide early whether you are writing the easy one or the
    # streaming one (chapter 10 section 4).
    pass


def softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    # TODO
    return out


# ----------------------------------------------------------- 4. layernorm
@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, stride, n_cols, eps,
                     BLOCK: tl.constexpr):
    # TODO
    pass


def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    out = torch.empty_like(x)
    # TODO
    return out
