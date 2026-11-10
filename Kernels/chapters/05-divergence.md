# Chapter 05 - Divergence and the warp

> The last foundations chapter, and the one that converts "32 threads execute
> together" from a fact you know into a thing you can exploit. Warp-level
> primitives introduced here are the engine of every reduction, scan and
> softmax in Part II.

---

## 1. One program counter, 32 lanes

A warp issues one instruction per cycle for all 32 lanes, with an **active
mask** saying which lanes commit their results. When lanes disagree about
control flow, the hardware executes both paths, masking alternately:

```cuda
if (threadIdx.x < 16) {
    A();      // lanes 0-15 active, lanes 16-31 masked (executing, discarding)
} else {
    B();      // lanes 16-31 active, lanes 0-15 masked
}
// reconvergence: all 32 active again
```

Time = time(A) + time(B). Not max. Both, always.

**Divergence is a property of a warp, not of a program.** This branch is free:

```cuda
if (blockIdx.x % 2 == 0) { A(); } else { B(); }
```

because every thread in a block -- hence every lane in every warp -- agrees.
Same for `if (threadIdx.x / 32 == 0)`, which splits at a warp boundary. The
expensive version splits *within* a warp:

```cuda
if (threadIdx.x % 2 == 0) { A(); } else { B(); }   // every warp diverges
```

Identical arithmetic, identical results, 2x the cost. The question is never
"does this code branch" but "**do the 32 lanes of a warp agree**".

### Costs, calibrated

| Pattern | Cost |
|---|---|
| All 32 lanes agree | free (just a predicate test) |
| Short divergent body (a few instructions) | ~free: compiler predicates it |
| Two long branches, 16/16 split | sum of both |
| `switch` with 32 different cases | up to 32x |
| Divergent loop trip counts | the warp runs until the *last* lane exits |

That last row is the sneaky one:

```cuda
for (int i = 0; i < count[tid]; ++i) work();
```

If one lane has `count = 1000` and the other 31 have `count = 1`, the warp
takes 1000 iterations with 31 lanes idle for 999 of them. Ragged per-thread
work is the classic GPU performance disaster, and it is why Chapter 22 (MoE,
variable-length sequences) is largely about load balancing.

### Predication

For short bodies the compiler does not branch at all:

```cuda
y = (x > 0) ? x : 0;      // SASS: FSETP + predicated FSEL. No branch.
```

Both values computed, one selected, no control flow. This is why `relu` and
masking are free and why clamping bounds inside a hot loop usually costs
nothing. In SASS you see `@P0` prefixes rather than `BSSY`/`BRA`/`BSYNC`.

---

## 2. Independent thread scheduling (Volta and later)

Pre-Volta, a warp had one PC and lanes were guaranteed to execute in lockstep
when not diverged. Programs exploited this with "warp-synchronous programming":
skipping `__syncthreads()` inside the last 32 elements of a reduction because
lockstep made it safe.

Volta gave every thread its own PC. Lanes can now be at different instructions
even without an explicit branch, and the hardware may reconverge whenever it
likes. Two consequences:

**1. Old warp-synchronous code is broken.** This, from every pre-2017 tutorial,
is now undefined behaviour:

```cuda
// WRONG on Volta+
if (tid < 32) {
    sdata[tid] += sdata[tid + 32];   // no barrier: assumes lockstep
    sdata[tid] += sdata[tid + 16];
    ...
}
```

The fix is explicit synchronisation or, better, shuffles:

```cuda
// Correct
if (tid < 32) {
    float v = sdata[tid];
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    ...
}
```

**2. Every warp primitive now takes a mask.** `__shfl_sync`, `__ballot_sync`,
`__any_sync` all require you to name the participating lanes. `0xffffffff` means
all 32. **The mask must match the lanes that actually reach the instruction** --
passing `0xffffffff` from inside a divergent branch is undefined behaviour,
and the usual symptom is a result that is right on one GPU and wrong on
another.

---

## 3. Warp-level primitives

These are why the warp is a feature and not just a tax. They exchange data
*between lanes, in registers*, with no shared memory and no barrier.

### Shuffles

```cuda
T __shfl_sync     (mask, var, srcLane)  // read lane srcLane's copy of var
T __shfl_up_sync  (mask, var, delta)    // read lane (me - delta)
T __shfl_down_sync(mask, var, delta)    // read lane (me + delta)
T __shfl_xor_sync (mask, var, laneMask) // read lane (me XOR laneMask)
```

A full warp reduction in five instructions, no shared memory, no barrier:

```cuda
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;   // lane 0 holds the total
}
```

Trace it: offset 16 makes lanes 0-15 hold pairwise sums; offset 8 makes lanes
0-7 hold sums of 4; after offset 1, lane 0 holds all 32. Five steps for 32
values, because `log2(32) = 5`.

Using `__shfl_xor_sync` instead gives a **butterfly**: every lane ends up with
the total, not just lane 0. Often what you want, same cost.

### Vote and ballot

```cuda
int  __all_sync(mask, pred);      // 1 if pred is true in every active lane
int  __any_sync(mask, pred);      // 1 if true in any
unsigned __ballot_sync(mask, pred); // bit i = pred of lane i
unsigned __activemask();          // which lanes are here right now
```

These enable *decisions made by the warp as a whole*:

```cuda
// Skip an expensive path if nobody needs it: the branch is now uniform.
if (__any_sync(0xffffffff, needs_work))
    expensive_path();
```

Ballot plus `__popc` gives you a warp-level count, and ballot plus a prefix
mask gives you a **stream compaction** offset in two instructions:

```cuda
unsigned active = __ballot_sync(0xffffffff, keep);
int rank  = __popc(active & lanemask_lt());  // how many kept lanes precede me
int total = __popc(active);
```

That pattern -- ballot, prefix-popcount, one atomic per warp for the base
offset -- is how every fast filter, compaction and histogram on a GPU is
written. It is Chapter 09.

### Warp-aggregated atomics

```cuda
// Naive: 32 atomics to the same address, fully serialised.
atomicAdd(counter, 1);

// Aggregated: one atomic per warp.
unsigned active = __ballot_sync(0xffffffff, true);
int leader = __ffs(active) - 1;
int total  = __popc(active);
int rank   = __popc(active & lanemask_lt());
int base;
if (lane == leader) base = atomicAdd(counter, total);
base = __shfl_sync(active, base, leader);
int my_slot = base + rank;
```

32x fewer atomics. The compiler now does this automatically for simple cases,
but you need to recognise the pattern because it generalises to cases the
compiler cannot see.

---

## 4. Barriers and the hazard

```cuda
__syncthreads();   // all threads in the BLOCK
__syncwarp(mask);  // the named lanes of one warp
```

**`__syncthreads()` must be reached by every thread in the block.** This hangs
or corrupts:

```cuda
if (tid < 100) {
    __syncthreads();    // threads >= 100 never arrive
}
```

The legal form puts the barrier outside the branch:

```cuda
if (tid < 100) { partial_work(); }
__syncthreads();
```

`compute-sanitizer --tool synccheck` finds these. The failure mode is
spectacular: a hang, or silently wrong results on a different GPU.

Cost: a barrier drains the block's pipeline. Every warp waits for the slowest.
In an inner loop, barriers are frequently the top stall reason -- which is the
entire motivation for double buffering in Chapter 12.

---

## 5. Divergence patterns in real kernels

**Bounds guards** -- `if (i < n)` -- diverge only in the last warp of the last
block. One warp out of thousands. Ignore it.

**The reduction tail** -- `if (tid < s)` halves active lanes each step. The
divergence is inherent to the algorithm; shuffles make the last 5 steps free,
which is why fast reductions switch to shuffles below 32 elements.

**Data-dependent work** -- `if (x[i] > threshold) expensive()`. The cost
depends entirely on how the data is *sorted*. Random data: every warp diverges,
cost = both paths everywhere. Sorted data: warps are uniform, cost = the true
average. **Sorting by branch condition before a divergent kernel is a real
optimisation**, and it is the core idea behind MoE token routing.

**Variable-length sequences** -- padding to the max wastes work proportional to
the variance of the lengths; sorting by length into buckets recovers it. Same
idea, different clothes.

---

## 6. Common failure modes

**Kernel hangs.** `__syncthreads()` inside divergent control flow.

**Right on one GPU, wrong on another.** Warp-synchronous assumptions without
`__syncwarp`, or a wrong mask in a `_sync` primitive.

**Slower on real data than on uniform test data.** Data-dependent divergence
you never saw in the synthetic test. Always benchmark on realistic
distributions.

**A `switch` that is catastrophically slow.** Lanes taking many different
cases. Can you sort or group so warps become uniform?

**Shuffle returns garbage.** The mask does not include the source lane, or the
source lane is inactive. A `__shfl_down_sync` reading beyond the last active
lane returns the caller's own value, which silently doubles entries in a
reduction.

**Loop is slow and divergence metrics look fine.** Check trip counts, not just
branches. The warp waits for its slowest lane.

---

## 7. Exercises

### 5.1 - Divergence accounting

`A()` takes 100 cycles, `B()` takes 200. For each, give the cycles per warp:

(a) `if (threadIdx.x < 32) A(); else B();` with `blockDim = 64`
(b) `if (threadIdx.x % 2) A(); else B();`
(c) `if (blockIdx.x % 2) A(); else B();`
(d) `if (data[i] > 0) A(); else B();` with uniformly random data
(e) the same as (d), but data sorted so positives come first
(f) `switch (i % 4) { ... }` with four 100-cycle cases

For (d), compute the *expected* cost per warp as a function of p = P(data > 0),
and find the p that is worst. Is it 0.5? Explain the shape of that curve.

### 5.2 - Trace a shuffle reduction

With `v = lane_id` in each of 32 lanes:

(a) Tabulate every lane's `v` after each of the 5 `__shfl_down_sync` steps.
(b) Which lanes hold meaningful values at the end? Which hold junk?
(c) Rewrite with `__shfl_xor_sync`. Tabulate again. Which lanes are meaningful
    now?
(d) The warp has only 20 active lanes (the rest exited). What does the
    `shfl_down` version produce, and what mask must you pass to make it
    correct? Is it even fixable, or does the algorithm need to change?

### 5.3 - Warp-level compaction

Using `__ballot_sync`, `__popc` and one atomic per warp, design (on paper) a
kernel that copies only the elements of `x` satisfying a predicate into a dense
output array, preserving order.

(a) How does a lane compute its output slot?
(b) How many atomics per warp? Per block? Could you reduce it further?
(c) Is the output write coalesced? Should you care?
(d) Compare to: one atomic per element. Estimate the speedup at 50% keep rate.
(e) What if the keep rate is 1%? 99%? Does your answer change?

### 5.4 - Find the bug

Each is broken or illegal. Say what breaks, when it manifests, and which
sanitizer tool would catch it.

```cuda
// (a)
if (tid < n) { partial(); __syncthreads(); }

// (b)
float v = x[tid];
if (tid % 2 == 0) v += __shfl_down_sync(0xffffffff, v, 1);

// (c)
for (int s = blockDim.x/2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
if (tid < 32) {                       // no barrier below this line
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
}

// (d)
unsigned m = __activemask();
v = __shfl_sync(m, v, 0);
```

For (d): it looks careful. Why is `__activemask()` a trap, and what should you
use instead?

### 5.5 - Predict the microbenchmark

Before running `05-divergence`: a kernel where each thread does `work_amount`
dependent FMAs, with a branch controlled by `threadIdx.x % divisor`.

(a) Sketch runtime vs `divisor` for divisor in 1,2,4,8,16,32,64.
(b) Where does the curve flatten, and why there?
(c) At `divisor = 64`, every warp is uniform. Is runtime back to baseline?
    Exactly? If not, what residual cost remains?
(d) Now make the branch bodies 3 instructions long instead of 300. Predict the
    new curve. What does the difference tell you about predication?

---

## Exercises to run

- [`05-divergence`](../exercises/05-divergence) - measure the real cost of
  divergence and match it against your predictions.

---

## What you should now believe

- Divergence costs the *sum* of the paths and is a property of the warp.
- A branch whose 32 lanes agree is free; predicated short bodies are free.
- Volta+ requires explicit masks and `__syncwarp`; lockstep is not guaranteed.
- Shuffles move data between lanes with no memory and no barrier, and they are
  the backbone of Part II.
- Ballot + popc + one atomic per warp is the compaction idiom; learn it now.
- Ragged per-thread work is the expensive kind of divergence, and sorting is
  often the fix.

Next: [Chapter 06 - Shared memory and tiling](06-shared-memory-tiling.md),
where you finally fix that transpose.
