# 16 - diagnose three slow kernels

**Chapter:** [16](../../chapters/16-profiling.md)

## Task

Three correct, slow kernels. Each is slow for a different reason. Diagnose each
**from profiler output before reading its source**, then fix it to reach 80% of
your copy ceiling.

## The rule

Open `kernel.cu` only after you have written your three diagnoses. The whole
skill being trained here is diagnosing a kernel you did not write and cannot
immediately read, which is what the job actually is. Reading the source first
converts a profiling exercise into a spot-the-bug puzzle and wastes it.

## Procedure

```bash
make run        # see how slow they are
ncu --section SpeedOfLight --section WarpStateStats \
    --section MemoryWorkloadAnalysis --section Occupancy ./exercise
```

Fill this in for each of the three, from counters only:

```
kernel 1:
  SOL memory ___%   SOL compute ___%   -> bound by: memory / compute / latency
  top stall: __________________ at ___%
  sectors per request: ___      bank conflicts: ___
  achieved occupancy: ___%      registers: ___
  DIAGNOSIS (one sentence): ______________________________________
  PREDICTED FIX: ________________________________________________
  PREDICTED SPEEDUP: ___x
```

Then, and only then, open the source, apply your fix, and verify that the
counter you targeted actually moved.

## Run

```bash
make run
make run ARGS=4096
```

## Pass bar

- Correctness: each fixed kernel must match its slow counterpart to 1e-4.
  (The slow versions are the reference -- they are correct, just slow.)
- Performance: all three fixed kernels >= 80% of copy ceiling.

## After it passes

1. **Verify every diagnosis.** For each kernel, report the counter you targeted
   before and after. If the time improved but the counter did not move, you
   fixed something else and you need to find out what.

2. **Rank your predictions.** How close were your predicted speedups? Where you
   were badly wrong, what did you not account for?

3. **Which was hardest to diagnose?** Usually it is the one whose SpeedOfLight
   numbers are both low, because "nothing is saturated" is less legible than
   "memory at 95%". Write down what you would look at first next time.

4. **Make your own.** Write a fourth slow kernel with a bottleneck of your
   choosing, hand it to someone else (or to yourself in a week) with no
   explanation, and see whether the counters give it away. Designing a
   diagnosable bug teaches you a surprising amount about which counters are
   actually informative.

<details>
<summary>If you are completely stuck after profiling (still no code hints)</summary>

- For a kernel whose `sectors per request` is 32: what has to be true about the
  addresses 32 lanes of one warp are touching? Which index in the kernel varies
  with `threadIdx.x`?
- For a kernel with a large `l1tex__data_bank_conflicts...` count: which shared
  access is it attributed to? What is the bank of that address for lane 0, lane
  1, lane 2?
- For a kernel whose top stall is `Barrier` and whose SOL numbers are both
  under 30%: what is every warp waiting for, how often, and does the
  computation actually require that wait?

</details>
