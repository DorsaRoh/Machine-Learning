# 23 - the Triton ladder

**Chapter:** [23](../../chapters/23-triton.md)

## Task

Rewrite four kernels you already wrote in CUDA -- vector add, reduction,
softmax, LayerNorm -- in Triton, and reach 80% of your own CUDA performance on
each.

The point is not that Triton is easier. It is to find out *exactly* what the
compiler gave you for free and what it charged you for.

## Setup

```bash
pip install triton          # bundled with recent PyTorch builds
cp cuda_times.example.json cuda_times.json
# then edit cuda_times.json with YOUR numbers from exercises 02, 07 and 10
python bench.py
```

The gate needs your CUDA times. Re-run `02-vector-add`, `07-reduce`,
`10-softmax` and `10-layernorm` at the shapes listed in the example file and
copy the medians across. Comparing against PyTorch alone would miss the
question this exercise asks.

## Predict first

For each of the four, before you write it:

```
kernel      CUDA ms    predicted Triton ms    why the gap (or why not)
add         ______     ______                 ______________________
sum         ______     ______                 ______________________
softmax     ______     ______                 ______________________
layernorm   ______     ______                 ______________________
```

## Pass bar

- All five correctness checks pass, including `softmax wide`.
- Each of the four within 80% of your CUDA version.

`softmax wide` is `[64, 131072]`. A row does not fit in registers at any
sensible `BLOCK`, so the one-shot `tl.load` of a whole row will fail. That is
deliberate: it is the same streaming-versus-one-pass decision as chapter 10,
and Triton does not make it for you.

## Keep a log

For each kernel record: time to write, lines of code, final performance, and
the one thing that surprised you. Chapter 23 exercise 23.1(e) asks for exactly
this table, and it is the output of this exercise that you will still be using
in a year.

## After it passes

1. **Sweep `num_warps`** (1, 2, 4, 8, 16) and `BLOCK` for each kernel. Plot.
   How much of your final number came from these two knobs rather than from
   anything you would call an algorithm?
2. **Read the SASS** for your Triton softmax (`cuobjdump -sass` on the cached
   cubin; `TRITON_CACHE_DIR` tells you where). Compare its inner loop against
   your CUDA version's. Where did the difference come from?
3. **Autotune** the layernorm with `@triton.autotune`. Does it find what you
   found by hand? How long did the first call take?
4. **Fuse.** Write `layernorm + gelu + residual` as one Triton kernel. Predict
   the speedup over three separate ops, then measure. This is where Triton
   earns its place.
5. **Interpret mode.** Break one kernel deliberately and debug it with
   `TRITON_INTERPRET=1`. Do it once now, so you know how when you need it.

<details>
<summary>Hints (questions only)</summary>

- `tl.arange(0, BLOCK)` is a vector in registers. How many registers per
  thread does `BLOCK = 4096` imply, and what does the compiler do when that
  exceeds the budget?
- `tl.sum(x, axis=0)` across a block: what did that compile to? Is it the
  ladder you wrote by hand in 07-reduce?
- For `total_sum`, who reduces the per-program partials? Is a second kernel the
  only option?
- For the wide softmax, what does the program need to carry across iterations,
  and have you seen that state before?
- `mask=` on a load: what value does a masked-out element take, and does that
  default work for a max reduction?

</details>
