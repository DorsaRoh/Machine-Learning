# Chapter 14 - Async pipelines: cp.async, TMA, warp specialization

> Tensor cores made the math 16x faster and left the memory system where it
> was. This chapter is about the three generations of hardware response to that
> imbalance, and about the one idea underneath all of them: **stop making warps
> wait for data**.

---

## 1. The problem, stated precisely

The Chapter 12 double-buffered loop still has a flaw:

```cuda
float tmp = A[global_idx];       // global -> register. The warp STALLS here.
__syncthreads();
As[buf][idx] = tmp;              // register -> shared
```

The data goes global -> register -> shared. The register write is a dependency,
so the warp genuinely waits ~500 cycles, holding registers, occupying an issue
slot's worth of attention. Prefetching further ahead means more registers held
longer, which means lower occupancy, which means less latency hiding. You are
pushing on a balloon.

---

## 2. `cp.async` (Ampere, sm_80+)

```cuda
#include <cuda_pipeline.h>

__pipeline_memcpy_async(&shared_dst, &global_src, 16);  // 4, 8 or 16 bytes
__pipeline_commit();                                     // close this stage
__pipeline_wait_prior(1);                                // wait until <=1 stage pending
__syncthreads();
```

or in PTX:

```
cp.async.cg.shared.global [%0], [%1], 16;
cp.async.commit_group;
cp.async.wait_group 1;
```

**Global to shared, bypassing registers entirely, asynchronously.** The warp
issues the copy and keeps going. No register held, no stall at issue.

Which unlocks **multi-stage pipelining**:

```
stage buffers: 3 or 4 deep in shared memory

prologue: issue copies for stages 0, 1, 2
loop k:
    wait for stage k mod S to arrive
    issue copy for stage k+S-1
    compute on stage k mod S
```

With 3-4 stages in flight you cover the full memory latency with *shared
memory* rather than registers, and registers stay free for accumulators. This
is the structure of every Ampere-generation GEMM and of FlashAttention-2.

Variants worth knowing: `.ca` caches in L1, `.cg` bypasses L1 (use this for
streaming tile data you will not re-read), and `cp.async.ignore-src` fills with
zeros for out-of-bounds elements -- which handles GEMM tails with no branches
at all.

### How deep?

```
stages needed = memory_latency / time_per_stage_compute
```

For ~500 cycles of latency and a K-step that takes ~150 cycles of math, you
need 3-4 stages. More stages cost shared memory and reduce blocks per SM. It is
a tuning parameter; 3 or 4 is almost always right.

---

## 3. TMA (Hopper, sm_90+)

The Tensor Memory Accelerator is a **hardware DMA engine for multidimensional
tiles**.

Instead of every thread computing addresses and issuing copies, you build a
descriptor on the host describing the tensor (base pointer, dimensions,
strides, tile shape, swizzle mode) and then **one thread** issues:

```
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_addr], [tensor_map, {x, y}], [mbarrier];
```

The TMA engine fetches the whole tile, handles boundary clamping, applies the
swizzle, and signals an mbarrier on completion.

What this buys:

- **Zero address arithmetic in the warps.** On an Ampere GEMM, a surprising
  fraction of issued instructions are address computation. TMA deletes them.
- **One thread issues, the rest compute.** Which leads directly to warp
  specialization below.
- **Hardware boundary handling.** Out-of-range reads return zeros. GEMM tails
  for free.
- **Hardware swizzling.** The bank-conflict problem is solved by the engine.

TMA is why an H100 GEMM looks structurally different from an A100 GEMM, and it
is the enabling piece of FlashAttention-3.

---

## 4. Warp specialization

Once copies are issued by one thread and completion is signalled through
barriers, warps no longer need to be interchangeable. Split them by role:

```
producer warps : issue TMA/cp.async loads, signal arrival barriers
consumer warps : wait on barriers, run mma.sync, signal empty barriers
```

A ring buffer of shared-memory stages mediates between them, with two barrier
arrays (`full` and `empty`) per stage. The producer waits for `empty`, fills a
stage, signals `full`. The consumer waits for `full`, computes, signals
`empty`. It is a textbook bounded-buffer producer-consumer, implemented with
`mbarrier` objects in shared memory.

Why it wins:

- **No block-wide barriers.** `__syncthreads()` makes everyone wait for the
  slowest; here, producers and consumers only synchronise through stage
  barriers, so neither blocks the other unnecessarily.
- **Specialised register allocation.** Hopper's `setmaxnreg` lets producer
  warps release registers they do not need to consumer warps that do. A
  producer needs almost none; a consumer wants 200+ for accumulators.
- **Better instruction mix per warp.** A warp doing only `mma` keeps the tensor
  pipe saturated; a warp alternating loads and math does not.

Hopper adds **thread block clusters**: groups of blocks guaranteed co-resident
on one GPC, able to read each other's shared memory ("distributed shared
memory"). This lets a tile loaded by one block be consumed by several, cutting
redundant loads -- a genuinely new level in the hierarchy, between block and
grid.

---

## 5. Which to use

| Hardware | Technique | Typical gain over Chapter 12 |
|---|---|---|
| Any | double buffer through registers | 10-20% |
| sm_80+ | `cp.async`, 3-4 stages | 20-40% |
| sm_90+ | TMA + warp specialization | another 20-30% |

If you are on a T4 (sm_75), `cp.async` does not exist and the register-staged
double buffer from Chapter 12 is your ceiling. That is fine; the concept
transfers and the exercise below has a non-Ampere path.

---

## 6. Common failure modes

**`cp.async` version is slower.** Too few stages (no overlap), or too many
(shared memory ate your occupancy). Sweep the depth.

**Results wrong intermittently.** A missing `__pipeline_wait_prior` /
`cp.async.wait_group` before reading the stage, or a missing `__syncthreads()`
after it. The async copy's completion is *not* implied by anything else.

**`cp.async` silently compiled to a normal copy.** It requires sm_80+ and sizes
of exactly 4, 8 or 16 bytes with alignment to match. Check the SASS for
`LDGSTS`.

**Deadlock in warp specialization.** Producer waiting on `empty` while the
consumer waits on `full`: an off-by-one in the barrier phase. Draw the state
machine, do not debug by editing.

**TMA descriptor errors.** The tensor map must be built correctly on the host,
be 64-byte aligned, and its dimensions must match what the kernel assumes. The
failure mode is usually garbage data rather than an error.

**Register pressure got worse, not better.** You kept the register-staged
prefetch *and* added `cp.async`. Remove the old path.

---

## 7. Exercises

### 14.1 - Pipeline depth arithmetic

Your GPU, a 128x128x32 fp16 GEMM:

(a) Time for one K-step of math (count the `mma` instructions and use your
    GPU's tensor throughput).
(b) Memory latency in cycles, and in units of (a).
(c) Stages needed to fully cover it.
(d) Shared memory per stage; total for that many stages.
(e) Blocks per SM at that footprint. Occupancy.
(f) Is the depth that covers latency the same as the depth that maximises
    throughput? If not, which wins and why?

### 14.2 - Draw the timelines

For a 4-iteration loop, draw load and compute activity over time for:

(a) single buffered
(b) double buffered through registers
(c) `cp.async` with 3 stages
(d) warp specialized with 4 stages

Mark, in each, where the memory system idles and where the math pipes idle.
Then compute the total time for each, assuming load = 500 cycles and compute =
200 cycles per tile. Which reaches the math roof, and at what pipeline depth?

### 14.3 - Convert your GEMM

If you have sm_80+:

(a) Replace the global -> shared path in your `11-gemm-ladder` v5 with
    `cp.async`, 2 stages. Measure.
(b) 3 stages. 4 stages. 5 stages. Plot.
(c) Check the SASS for `LDGSTS`. If it is not there, nothing you measured means
    anything.
(d) Profile: did `Stall Long Scoreboard` drop? Did `Stall Barrier` drop? Which
    one did you actually fix?
(e) Registers per thread, before and after. Did freeing the staging registers
    buy you occupancy, and did the occupancy buy you speed?

### 14.4 - Producer-consumer by hand

Without Hopper hardware, you can still build the structure with `cp.async` and
shared-memory barriers:

(a) Design the ring buffer: how many stages, what state per stage?
(b) Which warps are producers? What ratio?
(c) Write the state machine for both roles, including the phase bit that stops
    a consumer from re-consuming a stage.
(d) Where can it deadlock? Prove it cannot, for your design.
(e) Implement it. Does it beat your `cp.async` version? If not, why would the
    extra structure be worth it on Hopper but not here?

### 14.5 - Read FlashAttention-3

Open the FA3 paper and the corresponding kernel source.

(a) Find the producer warps. What do they do?
(b) Find the consumer warps. How are they split between the two GEMMs and the
    softmax?
(c) How does it overlap softmax (non-tensor-core work) with matmul
    (tensor-core work)? Why is that overlap worth engineering for?
(d) Where is TMA used, and where is it not?
(e) What does `setmaxnreg` do in it, and what would break without it?

This is the state of the art as of writing. You now have every concept needed
to read it.

---

## Exercises to run

- [`14-pipelined-gemm`](../exercises/14-pipelined-gemm) - beat your own v5 by
  1.15x with pipelining.

---

## What you should now believe

- The bottleneck after tensor cores is feeding them, and every modern feature
  addresses that.
- `cp.async` moves global to shared without registers, enabling deep pipelines.
- Pipeline depth is set by latency divided by per-stage compute time, bounded
  by shared memory.
- TMA removes address arithmetic and boundary handling from the warps.
- Warp specialization replaces block barriers with a producer-consumer queue
  and lets register allocation differ by role.
- On pre-Ampere hardware, register-staged double buffering is the ceiling.

Next: [Chapter 15 - Numerics and precision](15-numerics.md). Part III is done;
you can write a fast matrix multiply.
