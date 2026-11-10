# Chapter 04 - Occupancy, warps and registers

> Occupancy is the most over-used and least understood knob in CUDA. This
> chapter's goal is to make you stop treating it as a score to maximise and
> start treating it as one input to a latency budget.

---

## 1. What occupancy is

**Occupancy = resident warps per SM / maximum resident warps per SM.**

Resident means "has its registers allocated and can be issued from this cycle".
Not running, not queued -- resident. An SM with 64 warp slots and 16 resident
warps is at 25% occupancy.

What it buys you is **latency hiding**. When a warp stalls on a 500-cycle
memory load, the scheduler issues from another resident warp. More resident
warps means more candidates and fewer idle cycles.

What it does not buy you is anything else. Occupancy is not speed. It is the
*supply* of independent work; whether you need that supply depends on how much
latency you have to cover and how much parallelism each warp already contains
on its own.

---

## 2. The four limiters

Resident warps per SM is the **minimum** of four independent constraints.

```
1. hardware cap        max_threads_per_SM / 32
2. registers           regs_per_SM / (regs_per_thread * 32)
3. shared memory       floor(smem_per_SM / smem_per_block) * warps_per_block
4. block slots         max_blocks_per_SM * warps_per_block
```

Worked on an A100 (2048 threads/SM = 64 warps, 65536 registers/SM, 164 KB
shared/SM, 32 block slots), with a kernel using 256-thread blocks, 40
registers/thread, 16 KB shared per block:

```
hardware  : 64 warps
registers : 65536 / (40 * 32) = 51 warps  -> rounded down by allocation granularity
shared    : floor(164/16) = 10 blocks x 8 warps = 80 -> capped at 64
blocks    : 32 x 8 = 256 -> capped at 64
=> 48 warps (registers bind, after granularity) = 75% occupancy
```

Finding which constraint binds is the whole analysis. Fixing the wrong one
wastes an afternoon.

**Find yours without running anything:**

```bash
nvcc -arch=native -Xptxas -v -c kernel.cu
# ptxas info: Used 40 registers, 16384 bytes smem, 0 bytes spill stores
```

Those three numbers plus the table above tell you your theoretical occupancy
before the GPU is even involved.

### Granularity

Registers are allocated in chunks (8 registers per thread on recent
architectures, warp-granular). So 33 registers may cost you the same as 40.
Shared memory is allocated per block in 128-byte or 256-byte units. These
rounding effects produce the *step* shape you see when sweeping block sizes --
performance is flat and then drops off a cliff, rather than degrading smoothly.

---

## 3. Why more occupancy is not always better

The famous counterexample is Volkov's: a GEMM at 25% occupancy beating the same
GEMM at 100%. The reason is **instruction-level parallelism**.

```cuda
// Low ILP: every instruction depends on the previous one.
// One memory request in flight per thread.
float s = 0;
for (int i = 0; i < N; ++i) s += a[i] * b[i];

// High ILP: four independent chains.
// Four memory requests in flight per thread.
float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
for (int i = 0; i < N; i += 4) {
    s0 += a[i+0] * b[i+0];
    s1 += a[i+1] * b[i+1];
    s2 += a[i+2] * b[i+2];
    s3 += a[i+3] * b[i+3];
}
```

Little's Law does not care where the concurrency comes from:

```
work in flight = warps x independent-requests-per-warp
```

64 warps with 1 request each, and 16 warps with 4 requests each, deliver the
same concurrency. But the second version uses 4x fewer warps, so each thread
can afford 4x the registers -- which is exactly what a register-tiled GEMM
needs to get its arithmetic intensity up.

This is the trade that defines high-performance GEMM:

> Spend registers on per-thread work (raising ILP and arithmetic intensity),
> accept lower occupancy, come out far ahead.

**The rule of thumb:** memory-bound kernels with little per-thread work want
high occupancy (50%+). Compute-bound kernels with deep per-thread ILP can be
excellent at 25%. The way to tell them apart is to look at stall reasons, not
at the occupancy number.

---

## 4. Register pressure and spilling

Registers are a hard, finite resource: 65536 32-bit registers per SM, 255 max
per thread. Ask for too many and one of two things happens.

**Lower occupancy.** The allocator gives you what you asked and fits fewer
warps. Usually fine.

**Spilling.** The compiler runs out and moves variables to "local memory",
which is *global memory* with a per-thread address. A register access becomes a
400-cycle round trip.

```bash
nvcc -Xptxas -v ...
# ptxas info: Used 255 registers, 192 bytes cumulative stack size,
#             128 bytes spill stores, 128 bytes spill loads
```

**Any nonzero spill count inside a hot loop is an emergency.** Spills in setup
code are harmless. Tell them apart by looking for `LDL`/`STL` in the SASS of
your inner loop.

Causes, in order of frequency:

1. Tile sizes too big: a `float acc[8][8]` register tile is 64 registers right
   there, before anything else.
2. **Dynamically indexed local arrays.** This one surprises everyone:

```cuda
float buf[16];
buf[i] = x;          // i not known at compile time -> the WHOLE array spills
```

Registers are not addressable. If the compiler cannot resolve every index at
compile time, the array must live in memory. Fully unrolled loops with constant
indices keep it in registers; a single dynamic index moves all 16 values to
global memory.

3. Too much live state across a long region: compute values just before use.
4. `-maxrregcount` or `__launch_bounds__` set too aggressively.

### Controlling it

```cuda
__global__ __launch_bounds__(256, 4)   // 256 threads/block, >= 4 blocks/SM
void my_kernel(...) { ... }
```

This promises the compiler a minimum occupancy, so it caps registers to fit. It
*also* tells it it may spill to meet that promise. Measure both ways: the
version that spills less is not always the one that runs faster, and you should
know which you have.

---

## 5. Choosing a block size

```
- multiple of 32, always. A block of 100 threads wastes 28 lanes of a warp.
- >= 128 so an SM can hold several blocks and barriers do not idle it.
- <= 1024 (hardware).
- 256 is the right first guess, essentially always.
```

Larger blocks: more shared memory per block, cheaper block-wide cooperation,
but coarser scheduling granularity and longer barriers.

Smaller blocks: finer scheduling, less tail waste, but more block-launch
overhead and less shared memory available per block.

The honest answer is that it interacts with register allocation, shared
footprint and tail effects in ways no rule predicts. **Sweep it.** That is
exercise `04-occupancy-sweep`. Real libraries autotune this at build time for
exactly this reason.

### The tail effect

A grid of 65 blocks on a GPU that holds 64 concurrently runs in two waves, and
the second wave uses 1 of 108 SMs. Runtime nearly doubles for 1.5% more work.

Diagnosis: `blocks / (SMs * blocks_per_SM)` -- if the fractional part is small
and nonzero, you are paying for a nearly empty wave.

Fixes: size the grid to an exact multiple of the SM count and use a grid-stride
loop; or make blocks smaller so the quantum is finer; or use a persistent
kernel with a work queue.

---

## 6. Achieved vs theoretical

Theoretical occupancy is what the resource limits allow. **Achieved** occupancy
is the time-averaged reality, and it is lower when:

- The grid is too small to fill the machine.
- Blocks finish at different times (load imbalance), so SMs drain.
- The tail wave.
- Barriers: a block at `__syncthreads()` has all its warps idle.

```bash
ncu --section Occupancy ./exercise
# Theoretical Occupancy   75%
# Achieved Occupancy      31%     <- the gap is the interesting part
```

A large gap is a scheduling or imbalance problem, not a resource problem, and
adding registers back will not help.

---

## 7. How to actually use this

The decision procedure, replacing "maximise occupancy":

```
1. Is the kernel memory-bound (check the roofline, Chapter 01)?
   yes -> you need enough concurrency to saturate bandwidth.
          Check: achieved bandwidth vs ceiling.
          If low AND occupancy low -> raise occupancy OR raise bytes-in-flight
          per thread (vectorize, unroll). Either works.
   no  -> occupancy matters much less. Go to 2.

2. What are warps stalling on? (ncu WarpStateStats)
   Long Scoreboard  -> memory latency. More in flight: occupancy or ILP.
   Barrier          -> too many syncs, or imbalance. Occupancy will not help.
   Math Throttle    -> genuinely compute-bound. Occupancy will not help.
   Not Selected     -> you have plenty of warps. Occupancy is NOT the problem.

3. Did you spill? If yes, fix that before touching anything else.
```

"Stall Not Selected is high and I am still slow" means the scheduler has plenty
of candidates and the bottleneck is elsewhere. Adding occupancy to that kernel
is pure cargo cult.

---

## 8. Common failure modes

**Occupancy 12.5% and nothing you change helps.** Check shared memory per
block. One oversized `__shared__` array can cap you at one block per SM.

**Occupancy dropped after a small code change.** You crossed a register
granularity step. `-Xptxas -v` before and after.

**100% occupancy, 15% of peak bandwidth.** Not an occupancy problem.
Coalescing (Chapter 03), or a dependency chain, or you are latency-bound on
something else.

**Kernel got slower when you unrolled more.** Register pressure rose, warps
fell, or you started spilling. Measure all three.

**Theoretical 75%, achieved 20%.** Grid too small, imbalance, tail, or
barriers.

**Faster at block=128 than 256, slower again at 64.** Granularity and tail
interacting. This is normal, it is why sweeps exist, and it is why the answer
is machine-specific.

---

## 9. Exercises

### 4.1 - Occupancy by hand

Your GPU (use `00-hello-gpu` output). For each kernel compute resident warps,
name the binding limiter, and give the occupancy:

(a) 256 threads/block, 32 regs, 0 shared
(b) 256 threads/block, 64 regs, 0 shared
(c) 256 threads/block, 32 regs, 48 KB shared
(d) 1024 threads/block, 32 regs, 0 shared
(e) 128 threads/block, 128 regs, 8 KB shared
(f) 64 threads/block, 16 regs, 0 shared

For (f): occupancy may look fine. What *other* limit is this kernel going to
hit, and why is a 64-thread block usually a bad idea anyway?

### 4.2 - The ILP trade

A kernel's inner loop is a chain of dependent FMAs with a 4-cycle latency. The
SM has 4 schedulers, each able to issue one instruction per cycle.

(a) With 1 resident warp and a fully dependent chain, what fraction of issue
    slots are used?
(b) How many resident warps make the FMA pipes busy every cycle?
(c) Instead, one warp with 4 independent accumulators. Issue utilisation now?
(d) How many warps with 4-way ILP are needed to saturate?
(e) Each extra accumulator costs registers. Sketch occupancy vs ILP for a
    kernel with a fixed 64-register budget, and mark where total throughput
    peaks.

This is the calculation behind every register-tiled GEMM. Do it properly now
and Chapter 11 will feel inevitable instead of magic.

### 4.3 - Spot the spill

Which of these spill, and why?

```cuda
// (a)
float acc[4][4];
#pragma unroll
for (int i = 0; i < 4; ++i)
  #pragma unroll
  for (int j = 0; j < 4; ++j) acc[i][j] = 0;

// (b)
float acc[16];
for (int i = 0; i < n; ++i) acc[i % 16] += x[i];

// (c)
float acc[8][8];
#pragma unroll
for (int i = 0; i < 8; ++i)
  #pragma unroll
  for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];

// (d)
float window[32];
for (int i = 0; i < 32; ++i) window[i] = x[idx + i];
float m = 0;
for (int i = 0; i < 32; ++i) m = fmaxf(m, window[i]);
```

For (c), count the registers that tile needs and compare with your GPU's
per-thread budget. Does it fit? At what occupancy? What tile size would you
pick instead?

### 4.4 - Design the sweep

Before running `04-occupancy-sweep`, write down:

(a) Which block sizes you would test and why those.
(b) For a pure-bandwidth kernel, predict the shape of bandwidth vs block size.
(c) Predict where the peak is for your GPU, and name the mechanism that causes
    the drop on each side of it.
(d) How would the curve change for a kernel using 32 KB of shared memory per
    block? Predict the new peak.

### 4.5 - Build your own roofline

This is Exercise 1.4 from Chapter 01, now that you can write kernels. Design a
kernel with a tunable number of FMAs per loaded byte, sweep it, and plot
achieved FLOP/s against arithmetic intensity.

(a) How do you stop the compiler eliminating math whose result is unused?
(b) How do you keep the dependency chain from making you latency-bound rather
    than compute-bound at high intensity?
(c) How do you keep L2 out of the measurement at low intensity?
(d) Plot it. Is the measured knee where `P/B` predicts? If it is to the right,
    what does that tell you about the achievable P or B?

Keep the plot. You will use it to sanity-check every kernel you write from here
on.

---

## Exercises to run

- [`04-occupancy-sweep`](../exercises/04-occupancy-sweep) - find the real peak,
  then explain it with `-Xptxas -v` and the limiter arithmetic.

---

## What you should now believe

- Occupancy is a supply of latency-hiding work, not a score.
- Four limiters; find which one binds before changing anything.
- ILP substitutes for occupancy, and buying ILP with registers is usually the
  right trade for compute-bound kernels.
- Spills in a hot loop are an emergency; dynamic indexing into a local array
  causes them silently.
- The stall-reason section, not the occupancy number, tells you whether
  occupancy is your problem.

Next: [Chapter 05 - Divergence and the warp](05-divergence.md).
