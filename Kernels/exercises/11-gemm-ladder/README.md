# 11 - the GEMM ladder

**Chapters:** [11](../../chapters/11-gemm-i.md), [12](../../chapters/12-gemm-ii.md)

## Task

`C = A @ B` for 4096 x 4096 x 4096 fp32. Five versions. v5 must reach **70% of
cuBLAS**.

This is the hardest exercise in the course so far and the one that pays the
most. Budget several sessions. Everything in Part V is built from what you
learn here.

## The prediction table

Fill in the prediction row **before** writing each version, and the measured
row after. Keep this file.

| v | technique | predicted TFLOP/s | predicted % cuBLAS | measured | % cuBLAS | surprise? |
|---|---|---|---|---|---|---|
| 1 | naive | | | | | |
| 2 | shared tiling | | | | | |
| 3 | 1D register tile | | | | | |
| 4 | 2D register tile | | | | | |
| 5 | vectorized + tuned | | | | | |

For each version also record, from `nvcc -Xptxas -v`:

| v | registers | shared bytes | spills | occupancy | FMAs per shared load |
|---|---|---|---|---|---|
| | | | | | |

The second table is the one that explains the first.

## Run

```bash
make run
make run ARGS="2048 2048 2048"
make run ARGS="4096 4096 1024"    # skinny K
make run ARGS="1024 1024 4096"    # fat K
```

## Pass bar

- Correctness: 1e-3 relative against cuBLAS, every implemented version. Note
  this is a *tolerance*, not bit-exactness -- cuBLAS accumulates in a different
  order than you do and both answers are equally valid. Chapter 15.
- Performance: **v5 >= 70% of cuBLAS**.

The harness sets `CUBLAS_PEDANTIC_MATH` so the baseline is genuine fp32 and not
TF32. Comparing your fp32 kernel against a TF32 cuBLAS would be measuring the
wrong thing by a factor of 8.

## Working method

1. **v1 first, and measure it.** You need the baseline, and you need to see
   0.25 FLOPs/byte in practice.
2. **One change at a time.** If v3 is slower than v2, do not proceed. Find out
   why. Usually it is spills, or a loop the compiler would not unroll.
3. **Check `-Xptxas -v` after every version.**
   ```bash
   nvcc -arch=native -Xptxas -v -c kernel.cu 2>&1 | grep -E "registers|spill|smem"
   ```
   A spill in the inner loop will cost you more than the tiling gained.
4. **Profile when a prediction misses.** `make profile`, then
   `ncu --section SpeedOfLight --section WarpStateStats ./exercise`.
5. **Tune last.** Do not sweep tile sizes until the structure is right.
   Sweeping a bad kernel just finds the best bad kernel.

## Common walls, as symptoms

- *v2 only 2x over v1*: the inner loop is still reading global memory, or your
  tile load is uncoalesced, or L2 was already doing the work for v1.
- *v3 slower than v2*: spills, or the TM loop is not unrolled.
- *v4 at 30% of cuBLAS*: bank conflicts in the inner loop. Measure them.
- *v5 no better than v4*: your `float4` loads did not become `LDG.E.128`.
  Check the SASS.
- *v5 at 60% and stuck*: read Chapter 12 section 1 and build the stall budget.
  The remaining 10% has a name.

## After it passes

1. **Wave quantization sweep** (Chapter 12 exercise 12.4). N from 1024 to 8192
   in steps of 128. Plot. Explain the sawtooth.
2. **Autotune** (exercise 12.5). Template the tile parameters, prune, sweep.
   How much does one fixed configuration cost you across shapes?
3. **Skinny shapes.** Run `4096 4096 128` and `128 128 4096`. Your kernel will
   be bad at one of them. Which, and what would you build instead?
4. **TF32.** Re-run cuBLAS with `CUBLAS_TF32_TENSOR_OP_MATH` and compare. The
   gap you see is your motivation for Chapter 13.

<details>
<summary>Hints</summary>

- v2: with a 32x32x32 tile, how many threads use each element of the A tile?
  That factor is your traffic reduction. Does your measured speedup match it?
- v3: your TM accumulators -- are they `float acc[TM]` with a `#pragma unroll`
  loop and constant bounds? If any index is runtime-dependent, the array is in
  local memory.
- v4: the inner loop should load TM values of A and TN values of B into
  registers, then do TM*TN FMAs. If you are re-reading shared memory inside the
  TM*TN loop, you have written v3 twice.
- v5: when you store the A tile transposed, which thread writes which element,
  and is the *global* read still coalesced? The transpose happens on the way
  into shared, not on the way out of global.
- Bank conflicts: `ncu --metrics
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./exercise`

</details>
