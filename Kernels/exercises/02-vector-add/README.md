# 02 - vector add

**Chapter:** [02](../../chapters/02-first-kernel.md)

## Task

`c[i] = a[i] + b[i]` for 64M fp32 elements, at >= 80% of your measured copy
ceiling.

## Predict first

Fill this in before you compile:

```
n = 2^26 = 67,108,864 elements
bytes moved      = ______        (careful: is a write one byte of traffic or two?)
measured ceiling = ______ GB/s   (from 00-hello-gpu)
predicted time   = ______ ms
predicted GFLOP/s= ______
fraction of your GPU's fp32 peak = ______%
```

That last number should be somewhere around 1%. If it shocks you, re-read
Chapter 00 section 7. It is the correct answer, not a failure.

## Run

```bash
make run
make run ARGS=1000003     # a size that is not a nice power of two
```

## Pass bar

- Correctness: exact (floating-point addition is deterministic here).
- Performance: >= 80% of `copy_ceiling_gbs()`.

## Going from "correct" to "fast"

A naive one-element-per-thread version usually lands between 70% and 85%
depending on the GPU. If you are short of the bar, the levers in order of
effect are:

1. **Bytes in flight per thread.** One 4-byte load per thread is the minimum
   possible concurrency. Chapter 00 section 4.
2. **Grid size.** Is your grid an awkward multiple of the SM count? What does
   the last wave look like?
3. **Work per thread.** A grid-stride loop amortises index setup and lets you
   pick the grid independent of `n`.

Do them one at a time and record the bandwidth after each. The exercise is the
measurement, not the final number.

## After it passes

Run `make profile` and find:

- `sectors per request` for your loads. Is it 4?
- `gpu__dram_throughput...pct_of_peak`. How does it compare to the gate's
  percentage, and why are they different numbers?
- Achieved occupancy. Now change your block size to 1024 and then to 64. Does
  the bandwidth move? Does occupancy move? Are the two correlated the way you
  expected?

Then: write a version that reads `float4` instead of `float`. Measure. Did the
SASS change to `LDG.E.128` (`make sass | grep LDG`)? Did the time change? If it
did not, you were already at the roof, and that is a result worth recognising.

<details>
<summary>Stuck on performance? (questions, not answers)</summary>

- How many bytes does one thread have outstanding at the moment it stalls?
  Multiply by your thread count. Compare with the Little's Law figure for your
  GPU from Exercise 0.2.
- `ceil(n/256)` blocks on a 108-SM GPU: how many full waves, and how full is
  the last one?
- If each thread handled 4 consecutive elements, would the warp still be
  coalesced? What if it handled 4 elements strided by the grid size instead?
- Is `cudaMemset(d_c, ...)` in the timing loop? Should the write to `c` cost
  you a read as well? (Chapter 03 section 6.)

</details>
