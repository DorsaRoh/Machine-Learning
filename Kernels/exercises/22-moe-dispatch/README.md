# 22 - MoE dispatch

**Chapter:** [22](../../chapters/22-sparse-moe.md)

## Task

Route 4096 tokens to 2 of 8 experts and compute only the work that is needed.
The reference runs everything through everything, so there is a free 4x
available. Capture 70% of it.

## Predict first

```
dense-masked FLOPs              : ______ GFLOP
necessary FLOPs                 : ______ GFLOP   ratio: ______
gather bytes (step 4)           : ______ MB
scatter bytes (step 6)          : ______ MB
contiguous bytes per gather idx : ______        -> is the gather free?
time lost to those two steps    : ______ ms
best speedup you can hope for   : ______x
```

The last line is the one that matters. If the permutation overhead is 30% of
the dense time, your ceiling is 2.5x and the 2.8x bar is unreachable -- which
would mean your permutation design is wrong, not that the bar is.

## Run

```bash
make run
make run ARGS="4096 1024 8 2 0.0"     # T H E ktop skew
make run ARGS="4096 1024 8 2 0.7"     # a collapsed router
make run ARGS="4096 1024 64 2 0.0"    # many experts, few tokens each
```

## Pass bar

- Within 2e-2 of the dense-masked reference.
- >= 2.8x the dense-masked reference at the default shape, which is 70% of the
  theoretical `E/ktop` = 4x.

## Build it in pieces

Each step is testable alone, and debugging a permutation after the fact is
miserable. Verify as you go:

1. **Histogram** the expert ids. Check the counts against the router load line
   the harness prints -- it prints the truth for exactly this purpose.
2. **Exclusive scan** to offsets. Last offset + last count must equal
   `T * ktop`.
3. **Scatter** assignment indices. Verify it is a permutation: sort it and
   check it is `0 .. T*ktop-1`, and check every assignment landed in its own
   expert's range.
4. **Gather** rows. Compare a few rows against `x` directly.
5. **Grouped GEMM.** Start with a loop of `cublasGemmEx` and a `cudaMemcpy` of
   the counts. It works, it synchronizes, and it gives you a number.
6. **Combine.** Two slots per token land in the same output row.

Only after step 6 passes should you start removing the synchronization.

## After it passes

1. **Where did the time go?** Time each of the six steps. Which dominates?
   Almost everyone is surprised by the answer; write down your guess first.
2. **Kill the sync.** Replace the cuBLAS loop with your own grouped GEMM where
   each block binary-searches the offsets array. Measure what the
   synchronization was costing.
3. **Imbalance.** `skew=0.7`. Your speedup will drop. By how much, and does the
   dense reference care at all? Now fix it with a persistent kernel and a work
   queue (chapter 22 exercise 22.3).
4. **Many experts.** `E=64`. The dense reference gets 8x more expensive and
   yours should not. Does it? If not, what became the bottleneck?
5. **Capacity factor.** Add a cap of `1.25 * T * ktop / E` tokens per expert,
   dropping the overflow. How much does the regular shape buy, and how many
   tokens did you drop at `skew=0.7`?
6. **Determinism.** Is your combine deterministic? Check bit-for-bit across
   runs. If not, is that a design choice or an accident?

<details>
<summary>Hints (questions only)</summary>

- A token row is `H` halves. How many contiguous bytes is that, and what did
  chapter 22 section 3 say about gathers at that granularity?
- The histogram has `E = 8` bins and `T * ktop = 8192` contributors. Which of
  chapter 09's strategies fits that shape?
- Step 3 needs, for each assignment, a unique slot inside its expert's range.
  What primitive gives you "a unique index among everyone who picked the same
  bin"?
- In step 6, two slots write the same output row. Could you order the
  permutation so that they do not collide, instead of using an atomic?
- The grouped GEMM has `E` different row counts. Where do those counts live,
  and what is the cost of getting them to where cuBLAS needs them?

</details>
