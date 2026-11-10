# Chapter 16 - Profiling: Nsight and SASS

> From here on, a performance change you cannot explain with a counter does not
> count. This chapter turns "it got faster" into "it got faster because sectors
> per request went from 12 to 4".

---

## 1. The discipline

Guess-and-check works on small kernels and fails completely on big ones,
because the search space explodes and the interactions between changes stop
being intuitive. The replacement:

```
1. Predict:   from the roofline, what SHOULD this take?
2. Measure:   what does it take?
3. Localise:  which counter explains the gap?
4. Fix:       change exactly one thing that moves that counter.
5. Verify:    did that counter move? Did the time move with it?
```

Step 5 is the one people skip, and skipping it is how you accumulate a kernel
full of superstitions. If the time improved but the counter you targeted did
not move, you fixed something else and you do not know what.

---

## 2. Which tool

| Question | Tool |
|---|---|
| Is the GPU busy? Are kernels overlapping? Where are the gaps? | `nsys` |
| Why is *this kernel* slow? | `ncu` |
| Is it correct? | `compute-sanitizer` |
| What instructions actually run? | `cuobjdump -sass` |
| How many registers, how much shared, any spills? | `nvcc -Xptxas -v` |

Start at the top. Profiling a kernel that only accounts for 5% of your runtime
is a very productive-feeling waste of a day.

---

## 3. Nsight Systems: the timeline

```bash
nsys profile -o timeline --trace=cuda,nvtx,osrt ./app
nsys stats timeline.nsys-rep
```

What to look for:

- **Gaps between kernels.** The GPU is idle. CPU-bound launching, or
  synchronisation, or small kernels dominated by launch overhead.
- **Serialised copies and kernels.** They could overlap. Chapter 17.
- **One kernel dominating.** Now go to `ncu` on that one.
- **Many tiny kernels.** Fusion candidates (Chapter 10) or CUDA Graphs
  (Chapter 17).

Annotate your own phases so the timeline is readable:

```cuda
#include <nvtx3/nvToolsExt.h>
nvtxRangePush("attention");
...
nvtxRangePop();
```

Ten minutes of NVTX annotation makes a timeline self-explanatory for everyone
who looks at it afterwards, including you in a month.

---

## 4. Nsight Compute: the kernel

```bash
ncu --set full -o report ./exercise
ncu --section SpeedOfLight ./exercise
ncu --kernel-name my_kernel --launch-skip 5 --launch-count 1 --set full ./exercise
```

Always profile a **warmed-up** launch. The first one includes JIT, allocation
and page faults, and will slander your kernel.

### Read it in this order

**1. Speed Of Light.** Two numbers: % of peak memory throughput and % of peak
compute throughput.

```
Memory 85%, Compute 15%  -> memory-bound. Reduce bytes. You are near done.
Memory 10%, Compute 90%  -> compute-bound. Fewer/cheaper instructions.
Memory 20%, Compute 15%  -> LATENCY-bound. Not enough work in flight.
```

The third case is the one beginners misread as "balanced".

**2. Memory Workload Analysis.** The chart showing bytes moving between HBM,
L2, L1 and the SM. Compare each number against your hand count from Chapter 01.
A 3x discrepancy at the L1-to-L2 edge is a coalescing problem; at the L2-to-HBM
edge it is a reuse problem.

The key derived number:

```
sectors per request = l1tex__t_sectors_... / l1tex__t_requests_...
```

4 is perfect for fp32. 32 is the worst case. Anything above 4 is a
quantified coalescing opportunity.

**3. Warp State Statistics.** Why resident warps are not issuing. The full
table is in the [profiling cheat sheet](../reference/profiling-cheatsheet.md);
the short version:

- `Long Scoreboard` -> waiting on global memory
- `Short Scoreboard` -> waiting on shared memory / MIO
- `Barrier` -> waiting at `__syncthreads`
- `Wait` -> fixed-latency dependency, i.e. a dependency chain
- `Not Selected` -> ready but another warp went first. **This is good news.**

**4. Occupancy.** Theoretical vs achieved. A gap means scheduling, imbalance or
tail, not resources.

**5. Source counters.** With `-lineinfo` compiled in, Nsight attributes stalls
to source lines. This is where "the problem is line 47" comes from, and it is
worth the GUI.

### Useful invocations

```bash
# Just the two numbers that classify the kernel
ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed ./exercise

# Coalescing
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./exercise

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./exercise

# Tensor cores actually used?
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./exercise

# Compare two versions
ncu --set full -o before ./exercise_v1
ncu --set full -o after  ./exercise_v2
ncu --import after.ncu-rep --baseline before.ncu-rep --page details
```

That last one -- baseline comparison -- is the single most useful feature in
the tool and almost nobody uses it. It prints every counter with its delta.

---

## 5. Reading SASS

When the profiler's answer stops being specific enough:

```bash
nvcc -arch=native -lineinfo -o exercise main.cu
cuobjdump -sass exercise | less
nvdisasm -c -g exercise      # with control flow graph and line numbers
```

What to look for, in order:

| Instruction | Meaning | Why you care |
|---|---|---|
| `LDG.E.128` / `STG.E.128` | 128-bit global access | your `float4` worked |
| `LDG.E` (32-bit) | scalar global access | your `float4` did *not* work |
| `LDS` / `STS` | shared memory | count them in the inner loop |
| `LDSM` | `ldmatrix` | tensor-core operand feed |
| `HMMA` / `IMMA` | tensor core math | its absence answers "why no speedup" |
| `LDL` / `STL` | local memory = **spills** | find and eliminate |
| `BAR.SYNC` | barrier | count per inner-loop iteration |
| `@!P0` prefix | predicated execution | cheap branchless conditional |
| `BSSY` / `BSYNC` | real divergence | not free |
| `IMAD` in the inner loop | address arithmetic | competing with your FMAs |
| `FFMA` | the actual work | count the ratio of these to everything else |

A useful quick check on any hot loop: dump the SASS, find the loop body, and
compute `FFMA count / total instruction count`. If it is below 0.5, half your
machine is doing bookkeeping, and that is where your next win is.

```bash
# Register and shared usage, without running anything
nvcc -arch=native -Xptxas -v -c kernel.cu
# ptxas info : Used 96 registers, 16384 bytes smem, 0 bytes spill stores
```

---

## 6. Benchmarking honestly

Things that make a benchmark lie, all of which you have now seen:

1. **No warmup.** First launch pays context creation and JIT.
2. **Wall-clock around an async launch.** Measures the launch, not the kernel.
   Use CUDA events.
3. **L2 still warm.** Repeated runs over a small buffer. Flush it.
4. **Mean instead of median.** One descheduled iteration dominates.
5. **Clock throttling.** Long benchmark runs heat the GPU and clocks drop.
   Check `nvidia-smi -q -d CLOCK` or lock clocks with `nvidia-smi -lgc`.
6. **Dead code elimination.** The compiler deleted your computation because
   nothing used the result. Always write something derived from it.
7. **Different problem sizes.** Comparing your kernel at 4096 against cuBLAS at
   4096 is fair; comparing against a library doing a slightly different
   operation is not.
8. **Including H2D transfers in one version and not the other.**

`bench.cuh` handles 1-4 for you. 5-8 are on you, every time.

---

## 7. A worked diagnostic

A tiled GEMM runs at 2.1 TFLOP/s against a 19.5 peak. Walk the tree:

```
SpeedOfLight: memory 22%, compute 11%
  -> neither roof. Latency-bound.

WarpStateStats: Stall Barrier 45%, Stall Short Scoreboard 30%
  -> warps are waiting at __syncthreads and on shared memory.

Bank conflicts: 8.2 average conflicts per shared load
  -> the shared read in the inner loop is conflicting.
     That also explains the barrier stall: the warps that
     finish their conflicted loads late make everyone wait.

Fix: transpose the A tile in shared memory so the column read
     becomes contiguous.

Verify: conflicts -> 0.1. Short scoreboard -> 8%. Barrier -> 12%.
        Time 6.2 ms -> 3.4 ms.  SpeedOfLight compute now 34%.

Next iteration: barrier is still 12% -> double buffer.
```

Notice that the two symptoms (barrier and shared stalls) had **one** cause.
That is typical, and it is why you fix the counter rather than the symptom.

---

## 8. Common failure modes

**`ncu` fails with "permission denied".** Profiling counters need elevated
access. `sudo`, or `--target-processes all`, or on a cloud instance you may
need a different provider -- Colab blocks counters intermittently.

**Profiling is extremely slow.** `--set full` replays each kernel many times.
Use `--section SpeedOfLight` while iterating.

**Numbers differ between runs.** Clock throttling, or you profiled a cold
launch. Use `--launch-skip`.

**The profiler says memory-bound; your hand count says compute-bound.** Your
hand count assumed reuse you are not getting. Compare `dram__bytes` against
your compulsory count -- the ratio is your reuse failure.

**Everything looks fine and it is still slow.** Look at the grid size against
your SM count (wave quantization), and at the kernel duration against 5
microseconds (launch overhead). Then go to `nsys`.

---

## 9. Exercises

### 16.1 - Profile everything you have built

For `06-transpose` v3, `07-reduce` v7, `10-softmax`, and `11-gemm-ladder` v5,
record:

```
SOL memory %, SOL compute %
sectors per request (load, store)
bank conflicts per shared access
achieved vs theoretical occupancy
top 3 stall reasons with percentages
registers, shared, spills
```

Then for each: is the dominant stall the one you would have predicted? Where
you were wrong, work out why. This table is the single most useful artifact you
will produce in Part IV.

### 16.2 - Baseline comparison

Take any two versions of one kernel.

```bash
ncu --set full -o before ./v1 && ncu --set full -o after ./v2
ncu --import after.ncu-rep --baseline before.ncu-rep --page details
```

(a) Which counters moved most?
(b) Does the counter you intended to change account for the time difference?
(c) Find one counter that moved and that you did not expect. Explain it.

### 16.3 - Read your own SASS

For your `11-gemm-ladder` v5:

(a) Find the inner loop in the SASS.
(b) Count `FFMA`, `LDS`, `IMAD`, `BAR.SYNC` in it.
(c) FFMA as a fraction of total instructions. Is it above 0.5?
(d) How many `IMAD` instructions per FFMA? Could any be hoisted out of the
    loop?
(e) Any `LDL`/`STL`? If yes, stop and fix that first.
(f) Compare to a cuBLAS kernel's SASS (`cuobjdump -sass` on the cuBLAS .so is
    large but grep-able). What is its FFMA fraction?

### 16.4 - Build your benchmark checklist

Write a one-page checklist you would hand to a colleague for benchmarking a
kernel, covering all eight lies in section 6. Then audit your own earlier
exercises against it and note which ones you got wrong at the time.

### 16.5 - Diagnose three broken kernels

Exercise [`16-diagnose`](../exercises/16-diagnose) contains three kernels that
are correct and slow, each for a different reason. **You must diagnose each
from profiler output before reading its source.** The written diagnosis comes
first; the fix comes second.

For each: predicted bottleneck from counters alone, then the fix, then verify
the counter moved.

---

## Exercises to run

- [`16-diagnose`](../exercises/16-diagnose) - three broken kernels, each with
  its own bar.

---

## What you should now believe

- Predict, measure, localise, fix one thing, verify the counter moved.
- SpeedOfLight classifies the kernel in two numbers; low-low means
  latency-bound.
- Sectors per request is the coalescing meter; 4 is perfect.
- Stall reasons tell you what to fix; `Not Selected` dominating is good.
- SASS answers what the profiler cannot: spills, vectorization, tensor cores.
- Baseline comparison in `ncu` is the most underused feature in the toolchain.

Next: [Chapter 17 - Streams and concurrency](17-streams-concurrency.md).
