# 07 - the reduction ladder

**Chapter:** [07](../../chapters/07-reductions.md)

## Task

Sum 268M fp32 values. Seven versions, climbing from the textbook tree to
bandwidth. Set `IMPLEMENTED` in `kernel.cu` as you go; the harness runs every
version up to that number and prints the speedup chain.

This is the centrepiece exercise of Part II. Do not rush it, and do not skip
versions -- the chain of speedups *is* the lesson.

## The target

```
n = 2^28 = 268,435,456 fp32
compulsory traffic = ______ bytes
your ceiling       = ______ GB/s   (from 00-hello-gpu)
perfect time       = ______ ms     <- write this on a sticky note
```

## The table you are producing

Fill this in as you climb. The kernels are the means; this table is the
deliverable.

| v | what it fixes | predicted x | measured x | measured GB/s | % of ceiling | what v(n-1) was wasting |
|---|---|---|---|---|---|---|
| 1 | baseline | - | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |

Rule: **predict before measuring, every single time.** When the measurement
disagrees with your prediction by more than 30%, stop and find out why before
writing the next version. That investigation is where the learning is.

## Run

```bash
make run
make run ARGS=1048576     # small n: a different regime entirely
```

## Pass bar

- Correctness: relative error < 1e-4 against a float64 CPU reference. Note the
  reference is float64 deliberately -- your tree reduction should be *more*
  accurate than a naive sequential fp32 sum, not less.
- Performance: best version >= 85% of ceiling.

## Notes on the rungs

Chapter 07 section 3 states each step as a question. Do not read ahead to the
answer. What follows is only scaffolding for the harness, not hints.

- v1-v3 differ only in which threads are active at each tree step and which
  shared banks they touch. All three do the same additions.
- v4 changes how many elements a thread loads before the tree begins.
- v5 replaces the bottom of the tree. Volta rules apply: no implicit warp
  synchrony, use `__shfl_down_sync` with a correct mask.
- v6 is about what the compiler can do when a loop bound is a constant. You may
  need to change `reduce_v6` into a template and adjust the dispatch.
- v7 changes the ratio of memory work to tree work, and it is the only rung
  that can actually reach the bandwidth bar.

## After it passes

1. **The tree-vs-memory ratio.** For v6 and v7, count the shared-memory
   operations and the global loads per element. The ratio explains the entire
   final speedup. Write it out.

2. **Cross-block combination, three ways** (Chapter 07 exercise 7.5). Implement
   two-kernel, atomic, and last-block-finishes. Measure at n = 2^28 and at
   n = 2^20. Check bit-exact reproducibility over 100 runs for each.

3. **Beat or match CUB.**
   ```cuda
   #include <cub/cub.cuh>
   cub::DeviceReduce::Sum(temp, temp_bytes, d_in, d_out, n);
   ```
   Measure it. If CUB is faster, find the mechanism -- look at its block size,
   its elements per thread, and whether it vectorizes its loads. If you are
   faster, check you are not accidentally reading less data.

4. **Accuracy experiment.** Compare against a sequential fp32 CPU sum, not the
   float64 one. Which is closer to the float64 truth, yours or the CPU's? By
   how much? Explain with the `O(log n)` vs `O(n)` error-growth argument from
   Chapter 07 section 6.

5. **Generalise** (Chapter 07 exercise 7.6): max, argmax, sum+sumsq, and the
   `(max, sum-of-exp)` online-softmax pair. That last one is used verbatim in
   `10-softmax` and again in `19-flash-fwd`. Build it well here and you get the
   next two chapters cheaper.

<details>
<summary>Hints</summary>

- At tree step 1 of v1, which lanes of warp 0 are active? How many warps in a
  256-thread block have *any* active lane?
- v2 makes active threads contiguous. That fixes divergence. Check the shared
  addresses those contiguous threads touch -- did you fix divergence and create
  something else?
- In v4, with one element per thread, what fraction of threads does useful work
  in the first tree step? Where did the other half of your threads go?
- For v7: at n = 2^28 with a grid of `SMs * 4` blocks of 256 threads, how many
  elements does one thread sum? How many shared-memory tree steps run, total,
  for the whole kernel? Compare with v6.
- Is your grid-stride load coalesced? Trace lane 0 and lane 1 at iteration 0.

</details>
