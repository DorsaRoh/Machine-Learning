# 08 - exclusive scan

**Chapter:** [08](../../chapters/08-scan.md)

## Task

Exclusive prefix sum over 16M fp32, at >= 60% of your copy ceiling.

## The bar is the exercise

The harness prints two floors: a one-pass floor (read once, write once) and a
three-pass floor. The 60% bar sits between them, on purpose.

```
one-pass floor   : ______ ms
three-pass floor : ______ ms
60% bar          : ______ ms
```

Work out before you write anything whether a three-kernel block-sums scan can
clear the bar. If it cannot, you know what you are building.

## Suggested path

Build it in stages, and measure each one. You learn nothing from jumping
straight to the final version.

1. **Warp scan** with `__shfl_up_sync`. 32 elements, no shared memory, no
   barriers. Test it in isolation with a tiny n.
2. **Block scan** from warp scans: each warp scans itself, the warp totals get
   scanned, each warp adds its offset. Count your barriers -- you should need
   very few.
3. **Three-kernel full scan.** Correct, simple, and a useful measurement:
   confirm empirically that it lands near the three-pass floor.
4. **One pass.** Decoupled look-back (Chapter 08 section 5), or another
   single-pass design you can defend.

## Run

```bash
make run
make run ARGS=1048576
```

## Pass bar

- Correctness: 1e-4 relative against a float64 CPU scan. Inputs are positive,
  so the prefix sum is monotonic and there is no cancellation hiding errors.
- Performance: >= 60% of ceiling.

## If you implement look-back

This is a genuinely difficult piece of concurrent programming and it is the
first place in this course where the memory model bites. Things that will
happen to you:

- **It hangs.** A block is spinning on a predecessor that has not been
  scheduled. Assign tile indices with an atomic counter rather than using
  `blockIdx`, so dependency order matches scheduling order.
- **It is wrong one run in ten thousand.** A status flag became visible before
  the value it describes. You need the publish to be ordered: value first,
  then fence, then flag. Read Chapter 18 section on fences, and consider
  packing value and status into a single 64-bit atomic so the question cannot
  arise.
- **racecheck is quiet but it is still wrong.** `racecheck` sees shared-memory
  races, not global ordering bugs. Reason about this one; the tool will not
  save you.

Run it 10,000 times in a loop and diff the output against the reference every
time. A one-in-a-thousand bug is still a bug, and finding it this way is how
real kernels get validated.

## After it passes

1. **Compare to CUB.** `cub::DeviceScan::ExclusiveSum`. What fraction of CUB do
   you reach? CUB uses decoupled look-back, so if you implemented three-kernel,
   the gap is the cost of the extra passes -- confirm the number matches your
   byte count.
2. **Hillis-Steele vs Blelloch at block level** (Chapter 08 exercise 8.4).
   Implement both block scans, measure, and explain which wins on a GPU and
   why work-efficiency arguments mislead here.
3. **Stream compaction** (exercise 8.7). Build it on your scan and measure at
   keep rates of 1%, 50%, 99%.
4. **Segmented scan** (exercise 8.6). Define the operator, prove it
   associative, then implement.

<details>
<summary>Hints</summary>

- `__shfl_up_sync(mask, v, d)` for lanes with `lane < d` returns the lane's own
  value. What do you have to do about that?
- Inclusive to exclusive: a shift. Which direction, and what fills the hole?
  Can you get exclusive directly without a separate shift pass?
- In the three-kernel version, which arrays are read and written in each
  kernel? Add up every byte. Does your measurement match?
- For look-back: what does a block need to know before it can produce final
  output? What is the minimum it must wait for? Can it do useful work while
  waiting?
- Does your kernel read `in` more than once? If yes, that is a 1.5x penalty
  right there.

</details>
