# p4-math

**Chapter:** [P7 - The math](../../p7-the-math.md)

## Task

Write every operation the main course will spend six months accelerating:
dot product, matmul, matmul-with-a-transposed-operand, softmax, LayerNorm,
RMSNorm, and finally attention. Slowly and correctly, on a CPU. Two hours.

## Run

```bash
make run
```

## Pass bar

All 24 checks pass.

## Notes

- **`matmul_bt` must not build a transpose.** `B` is already stored as `N x K`;
  the only thing that changes is how you index it. That single realisation is
  why cuBLAS takes `transa`/`transb` flags rather than offering you a transpose
  function, and why Chapter 11's GEMM ladder has four variants.
- **The identity tests are doing real work.** `A @ I == A` and `I @ A == A`
  catch different bugs: swapping two index expressions passes one and fails the
  other. If only one fails, you know exactly which pair you swapped.
- **`layernorm_rows` is checked on a constant row**, where the variance is
  exactly zero. Without `eps` inside the square root you divide by zero and get
  NaN. That is not a contrived test; it happens on real data at initialisation.
- **`attention` is the final boss**, and it is forty lines. Build it from the
  pieces you already have: `matmul_bt` gives you the scores, `softmax_rows`
  normalises them, `matmul` applies them to `V`.
- The `dot(1/i, i)` check has a loose tolerance because `1.0f/(i+1)` is not
  exactly representable. The true answer is exactly 1000 and no correct
  implementation will produce exactly 1000. P4 arriving uninvited, as it does.

## The number to remember

Your `attention` allocates an `S x S` score matrix. At `S = 8192` with fp16,
that is **128 MB per head**, and a model has dozens of heads per layer and
dozens of layers. The scores are written to memory and read straight back,
used once.

FlashAttention's entire contribution is: *don't do that.* Keep a running max
and a running sum, process `K`/`V` in tiles, never materialise the matrix. You
will build it in [Chapter 19](../../../chapters/19-flash-attention.md). The
reason it is a famous paper and not a footnote is the memory you just
allocated.

## After it passes

1. Time your `matmul` at 512x512x512. Compute GFLOP/s using your own
   `matmul_flops`. Compare with the peak you estimated in P8.2. What fraction
   of your machine are you using? (Expect something humbling, like 2%.)
2. Swap the loop order of `matmul`'s three loops — all six permutations. Time
   each. The spread will be large. Which is fastest, and does the winner match
   what P3 section 5 predicts about the innermost loop?
3. Implement a tiled `matmul`: 64x64 blocks, so the working set fits in L1.
   How much faster? You have just hand-built the CPU version of Chapter 06.
4. Make `softmax_rows` a single pass over the data using the online update from
   P7 section 4 — track the running max and running sum together, rescaling the
   sum when the max changes. It must agree with your three-pass version to
   1e-6. This is the actual FlashAttention recurrence, and doing it here, on a
   CPU, with a debugger, is far easier than doing it for the first time inside
   a CUDA kernel.
5. Write the backward pass of `softmax_rows` by hand and check it against a
   finite-difference approximation. Chapter 20 will assume you have.

<details>
<summary>Hints (questions only)</summary>

- `matmul`: which index does `A` not use? Which does `B` not use? Which does
  `C` not use? Three loops, three index variables, and each operand ignores
  exactly one.
- `matmul_bt`: in `A @ B^T`, element `(m, n)` is the dot product of row `m` of
  `A` with which row of `B`?
- `matmul_flops`: per output element, how many multiplies and how many adds?
  How many output elements?
- `layernorm_rows`: is `eps` added before or after the square root? Try it the
  other way and see which the constant-row check accepts.
- `rmsnorm_rows`: what exactly did you remove relative to LayerNorm? How many
  passes over the row does each need?
- `attention`: for query `i`, which rows of `K` contribute when `causal` is
  true? What should the softmax denominator be in that case — is it still over
  all `S` keys?
- Causal attention at `i = 0`: one key, one score, softmax of a single number.
  What must the output be? Check that by hand before running anything.

</details>
