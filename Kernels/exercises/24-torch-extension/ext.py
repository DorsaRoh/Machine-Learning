"""Exercise 24 - the integration layer.

Compiling the kernel is the easy part. Making it a first-class PyTorch op --
one that torch.compile can trace through without breaking the graph -- is the
part that decides whether your speedup survives into a real model.

Chapter: ../../chapters/24-shipping.md section 1
"""

import os

import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))

# Compiles on first import, caches in ~/.cache/torch_extensions.
_mod = load(
    name="fused_ln_gelu_res_ext",
    sources=[os.path.join(_here, "kernel.cu")],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=False,
)


# ---------------------------------------------------------------- the op
#
# TODO: wrap _mod.fused_ln_gelu_res as a torch.library.custom_op named
# "kernels::fused_ln_gelu_res".
#
# Questions to answer in the process, because each has a wrong answer that
# works until it does not:
#   - does this op mutate any of its arguments? what do you pass to
#     mutates_args, and what does torch.compile do with that information?
#   - what are the type annotations, and why does the op registration need
#     them at all?
#
# TODO: register the fake ("meta") implementation. It must compute the output
# shape, dtype, device and layout WITHOUT touching data -- it runs on meta
# tensors, where there is no data to touch.
#
# Without it, bench.py's fullgraph check will fail, which is the point: a
# custom op with no fake impl is a graph break, and a graph break around your
# op can cost more than your kernel saves.


def fused(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
          eps: float = 1e-5) -> torch.Tensor:
    """What the harness calls. Point it at your registered custom op, not at
    _mod directly -- calling the raw extension works and defeats the exercise."""
    raise NotImplementedError


# ------------------------------------------------------------ the baseline
def eager(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
          eps: float = 1e-5) -> torch.Tensor:
    """Reference and baseline. Do not edit."""
    import torch.nn.functional as F
    return x + F.gelu(F.layer_norm(x, (x.shape[-1],), w, b, eps),
                      approximate="tanh")
