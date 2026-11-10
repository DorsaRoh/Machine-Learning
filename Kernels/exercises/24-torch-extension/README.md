# 24 - ship one kernel

**Chapter:** [24](../../chapters/24-shipping.md)

## Task

Turn `out = x + gelu(layernorm(x) * w + b)` into a single CUDA kernel, register
it as a real PyTorch custom op, and beat eager by 1.3x on a workload that looks
like a model rather than a microbenchmark.

Three files are yours: `kernel.cu` (the kernel and its binding) and `ext.py`
(the op registration). `bench.py` and `test_ext.py` are the harness.

## Predict first

```
eager: tensor passes over x        : ______    bytes ______ MB
yours: tensor passes over x        : ______    bytes ______ MB
your copy ceiling                  : ______ GB/s
lower bound on your time, 24 layers: ______ ms
predicted speedup vs eager         : ______x
predicted speedup vs torch.compile : ______x    <- think about this one
```

The last line is the interesting prediction. `torch.compile` also fuses this
chain. Write down what you expect *before* you see the number.

## Run

```bash
python test_ext.py      # correctness: edge shapes, odd inputs, numerics
python bench.py         # the gate
```

First run compiles the extension, which takes a minute.

## Pass bar

- Matches eager to 1e-4, including on a distribution with mean 1000.
- `torch.compile(..., fullgraph=True)` traces through your op with no graph
  break. This is a hard requirement, not a nicety -- see chapter 24 section 1.
- >= 1.30x eager on the 24-layer chain.
- `test_ext.py` fully green.

`test_ext.py` is where most people lose an afternoon, and that is the lesson:
PyTorch hands you non-contiguous slices, transposed views and 3D tensors as a
matter of course, and your microbenchmark never did.

## Order of work

1. Kernel, correct, for contiguous 2D fp32. `bench.py` correctness only.
2. The `TORCH_CHECK`s and the `empty_like`. Run `test_ext.py` and watch it
   fail; decide for each failure whether to support the case or reject it
   loudly. Both are legitimate; silently returning garbage is not.
3. The custom op and the fake implementation. `fullgraph` check.
4. Performance. You are memory-bound, so chapter 03 and chapter 10 are the
   whole story: `float4` loads, one pass over `x`, `w` and `b` cached.
5. Re-run the whole thing and read the p05/p95 spread, not just the median.

## After it passes

1. **The backward.** Chapter 20 section 3 for the LayerNorm part. Wrap in
   `torch.autograd.Function`, verify with `torch.autograd.gradcheck` in
   float64, then train something small with it and confirm the loss curve
   matches.
2. **Break your own benchmark.** Chapter 24 exercise 24.2, all five ways.
   Record the speedup each wrong method reports. Keep the table.
3. **bf16 and fp16.** Add dtype support. Accumulate in fp32 regardless. Does
   the speedup double? Should it?
4. **Package it.** A `setup.py` with `CUDAExtension`, multiple
   `-gencode` flags plus a PTX fallback, and `pip install -e .`.
5. **Put it in a real model.** nanoGPT or similar: replace the corresponding
   subgraph and measure step time. Report the end-to-end number even though it
   is smaller than this one. Especially because it is smaller than this one.
6. **CUDA Graphs.** Capture the 24-layer chain. How much was launch overhead?

<details>
<summary>Hints (questions only)</summary>

- How many times does your kernel read `x`? Can it be one? What do you have to
  hold to make it one, and does a row fit?
- `w` and `b` are `cols` floats read by every block. Where should they be?
- The shifted-mean test: which of the three variance algorithms from chapter 15
  survives mean 1000? Which survives mean 1e6?
- `register_fake` runs on meta tensors. What information is available to it,
  and what is not?
- A non-contiguous input: what are your two options, and what does each cost in
  the benchmark you are about to report?

</details>
