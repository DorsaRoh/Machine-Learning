# 09 - histogram

**Chapter:** [09](../../chapters/09-histograms-atomics.md)

## Task

Histogram 67M bin indices into 256 bins. The harness runs three distributions:
uniform, half-concentrated, and 90%-concentrated in a single bin.

**The performance gate is on the worst distribution.** Being fast on uniform
data is not the job; not collapsing on skewed data is the job.

## Predict first

```
                    predicted GB/s    % of ceiling
uniform                 ______           ______
p = 0.5 in bin 0        ______           ______
p = 0.9 in bin 0        ______           ______

predicted collapse factor (uniform / worst) : ______x
```

Predict these for a naive global-atomic version *first*. Then predict them
again for the version you intend to write. The difference between the two sets
of numbers is the value of the chapter.

## Run

```bash
make run
make run ARGS="67108864 1024"    # 1024 bins
make run ARGS="67108864 32"      # 32 bins: much worse contention
```

The 32-bin run is the interesting one. Fewer bins means more threads per bin
means more contention, even with uniform data.

## Pass bar

- Correctness: exact counts on all three distributions. These are integer
  counts; there is no tolerance and no excuse.
- Performance: the **worst** distribution must reach >= 50% of ceiling.

## The actual exercise

1. **Measure the naive version first.** Write the one-line global-atomic
   kernel, record all three numbers, and keep them. Everything after is
   measured against this.

2. **Privatize.** Shared histogram per block, merged at the end. Record the
   three numbers again. Which distribution improved most? Did the skewed case
   improve as much as you expected, or is shared-memory contention now the
   bottleneck?

3. **Aggregate.** Add `__match_any_sync` warp aggregation. Record again. This
   should barely help uniform data and transform the skewed case. Does it?

4. **Replicate.** R copies of the histogram in shared memory. Sweep R = 1, 2,
   4, 8, 16. Plot worst-case bandwidth against R, and plot occupancy against R
   on the same axes. Where do they cross, and is the optimum where you expected?

5. **The skew curve** (Chapter 09 exercise 9.5). Sweep p from 0 to 1 in steps
   of 0.1 for each strategy and plot all the curves together. This single plot
   contains the whole chapter.

6. **32 bins.** Re-run everything at `nbins = 32`. Something that worked at 256
   will struggle here. Explain why, and fix it.

## After it passes

- Profile the skewed run and find where the time goes. Is it the atomics, or
  is it now just memory?
- Compute the theoretical atomic count for each strategy and compare against
  your measured times. Does a simple "serialised atomics" model predict the
  numbers?
- Implement sort-then-count for the 90% case and compare. At what skew does
  sorting win? Does it ever, at 256 bins?

<details>
<summary>Hints</summary>

- With 90% of elements in bin 0, how many lanes of a typical warp target bin 0
  simultaneously? What does one `atomicAdd` from that warp cost?
- `__match_any_sync(0xffffffff, bin)` gives each lane the mask of lanes with
  the same bin. What do you do with that mask? Who issues the atomic, and with
  what value?
- Zeroing the shared histogram: 256 bins, 256 threads. What if nbins > blockDim?
  What must come after the zeroing, and what must come before the merge?
- With R replicas, which replica should thread `t` use? Why `t % R` rather than
  `t / R`? (Think about which threads of a *warp* collide.)
- Is `in` read coalesced? At one element per thread with a grid-stride loop,
  yes. Check it anyway.

</details>
