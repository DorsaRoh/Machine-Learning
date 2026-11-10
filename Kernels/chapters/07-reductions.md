# Chapter 07 - Reductions

> Sum a billion numbers to one. It sounds trivial and it is the best teaching
> problem in GPU computing, because the naive version is 50x off the roof and
> every one of the seven steps to the roof teaches a different lesson.
>
> This chapter deliberately does **not** show you the ladder. It shows you the
> constraints and makes you climb.

---

## 1. The problem and the bound

```
out = x[0] + x[1] + ... + x[n-1]
```

For n = 256M fp32:

```
W = n FLOPs = 2.7e8
Q = 4n bytes = 1.07e9       (read every element exactly once; write 4 bytes)
I = 0.25 FLOPs/byte
```

Hopelessly memory-bound. Therefore:

> **A perfect reduction runs at copy bandwidth.** Its time is `4n / B`. Any
> reduction slower than that is wasting memory bandwidth, and any "optimisation"
> that does not reduce memory traffic or improve its efficiency is irrelevant.

On a 2 TB/s GPU that is 535 microseconds for 256M floats. That is your target.
Write it down before you start.

---

## 2. Why it is hard

Addition is associative and commutative, so the order is free -- any binary
tree of additions gives (nearly) the same answer. The difficulty is not the
math, it is that **the GPU has no cheap global synchronisation**.

```
threads in a warp   : can exchange via shuffle, no barrier.          cheap
threads in a block  : can exchange via shared memory + barrier.      cheap
blocks in a grid    : cannot synchronise.                            expensive
```

A reduction fundamentally needs every value to reach one place. The structure
of every fast reduction follows from that single constraint:

```
level 1: each thread reduces its own elements      (registers, free)
level 2: reduce within a warp                      (shuffles, no barrier)
level 3: reduce across warps in a block            (shared memory + barrier)
level 4: combine across blocks                     (<- the expensive one)
```

Level 4 has three options, and choosing among them is a real engineering
decision:

- **Two kernel launches.** Kernel 1 produces one partial per block; kernel 2
  reduces the partials. Simple, deterministic, costs an extra launch and an
  extra memory round trip for the partials.
- **Atomics.** One `atomicAdd` per block to a global accumulator. One launch,
  but **non-deterministic** in floating point because the order varies run to
  run. For `gridDim` in the thousands, contention is low and this is fast.
- **A grid-wide barrier** via cooperative groups, or an atomic-counter "last
  block does the final reduction" trick. One launch, deterministic, more
  machinery.

Know all three. The two-kernel version is the right default; the atomic version
is the right answer when determinism is not required and launch overhead
matters.

---

## 3. The ladder you are going to climb

Mark Harris's classic deck optimises a reduction in seven steps, 30x total.
You are going to rediscover it. Here are the *constraints*, in the order they
bite. Each one costs you a specific factor, and finding it yourself is the
exercise.

**Step 1 - interleaved addressing with modulo.** The textbook tree: at step
`s`, threads where `tid % (2*s) == 0` add `sdata[tid + s]`. It works. Ask: how
many lanes of each warp are active at step 1? At step 2? And `%` on the GPU is
a multi-instruction sequence.

**Step 2 - interleaved addressing without modulo.** Same tree, index computed
as `2*s*tid` so the active threads are contiguous. Ask: what does this do to
*which* threads are active, and therefore to divergence? And what does it do to
which shared-memory *banks* those threads touch?

**Step 3 - sequential addressing.** At step `s` (halving from `blockDim/2`),
thread `tid < s` adds `sdata[tid + s]`. Ask: banks now? Divergence now?

**Step 4 - first add during load.** Each thread loads two elements and adds
them before the tree starts. Ask: with one element per thread, what fraction of
your threads do useful work in the very first tree step? What is the cheapest
way to fix that?

**Step 5 - unroll the last warp.** Below 32 active threads, the tree is
entirely within one warp. Ask: what is now unnecessary? (And note Chapter 05:
the pre-Volta version of this trick is *illegal* today. Use shuffles.)

**Step 6 - full unroll with templates.** Block size as a template parameter
makes every loop bound a compile-time constant. Ask: what does the compiler do
with that which it cannot do otherwise?

**Step 7 - multiple elements per thread (grid-stride).** Instead of one
element per thread, a fixed grid where each thread sums many. Ask: what is the
ratio of tree-work to load-work in each version? This is the step that finally
reaches bandwidth, and understanding *why* is the most important thing in the
chapter.

The arc, stated once: **the tree is not the work. Reading memory is the work.**
Early versions spend most of their time on a `log n` tree over one element per
thread. The final version spends nearly all its time streaming memory, with a
tiny tree at the end. That reframing is the lesson.

---

## 4. Tools you have

### Warp reduction with shuffles

```cuda
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;   // lane 0 holds the result
}
```

No shared memory, no barriers, 5 instructions. **Every fast block reduction is
built on this.** The standard two-level structure:

```
1. each warp reduces itself with shuffles            -> 1 value per warp
2. lane 0 of each warp writes to shared[warp_id]     -> <= 32 values
3. __syncthreads()
4. warp 0 reduces those <= 32 values with shuffles   -> block total
```

One barrier for the whole block reduction. Compare that with a shared-memory
tree, which needs `log2(blockDim)` of them.

### Atomics

```cuda
atomicAdd(&global_sum, block_total);   // one per block
```

Fast when `gridDim` is in the thousands and each block contributes once. Lethal
if every thread does it.

**Determinism warning:** `atomicAdd` on floats combines in arbitrary order, and
floating-point addition is not associative. Two runs can differ in the last
bits. For a loss value, fine. For a regression test with a tight tolerance, or
for reproducible training, not fine. Chapter 15 discusses what to do about it.

### `cub`

```cuda
#include <cub/cub.cuh>
cub::DeviceReduce::Sum(temp, temp_bytes, d_in, d_out, n);
```

CUB is NVIDIA's tuned building-block library and it reaches ~95% of bandwidth.
In production, use it. Here, beat it or at least reach it, because the point is
to learn where its performance comes from.

---

## 5. Beyond sum

The same skeleton reduces anything associative. Only the operator changes:

```cuda
template <typename Op>
__device__ float warp_reduce(float v, Op op) {
    for (int off = 16; off > 0; off >>= 1)
        v = op(v, __shfl_down_sync(0xffffffff, v, off));
    return v;
}
```

- **max / min**: identity `-inf` / `+inf`. Used in softmax.
- **argmax**: reduce a `(value, index)` pair. Tie-breaking rules matter;
  decide and document.
- **sum + sum-of-squares**: two accumulators in one pass. This is LayerNorm,
  Chapter 10. Beware numerical stability (Welford, Chapter 15).
- **online softmax**: the operator is `(max, sum)` with a rescaling rule. It is
  still associative, and that fact is what makes FlashAttention possible.
  Chapter 10 derives it.

If you can prove your operator is associative, this skeleton works. If you
cannot, you cannot reduce it in parallel -- that is the actual requirement, not
a detail.

---

## 6. Numerical accuracy, free

Summing 256M fp32 values sequentially loses precision badly: once the running
total is large, small addends vanish into rounding. Relative error grows like
`O(n * eps)`.

A tree reduction has depth `log n`, so error grows like `O(log n * eps)`.

```
n = 2^28, eps = 1.2e-7
sequential : ~32      relative error -- catastrophic, result is garbage
tree       : ~3.4e-6  relative error -- fine
```

**The parallel version is more accurate than the serial one**, by a factor of
`n / log n`. This is worth knowing because it is counterintuitive and because
it means a GPU reduction disagreeing with a NumPy reduction is often the GPU
being *right*. (NumPy uses pairwise summation for exactly this reason.)

---

## 7. Common failure modes

**Result is wrong by a factor of 2, or missing a chunk.** Tree indexing, or the
last partial block, or forgetting the final cross-block combine.

**Wrong only for n not a power of two.** Padding with the identity element, or
guarding the tree. Decide which and apply it consistently.

**Intermittently wrong.** Missing `__syncthreads()` between tree steps, or
warp-synchronous code without `__syncwarp` (Chapter 05).

**Result differs slightly between runs.** Atomics. Expected. Decide whether you
care.

**Fast for small n, slow for large n.** Not enough elements per thread -- you
are tree-bound, not memory-bound. Step 7.

**At 20% of bandwidth with a perfect tree.** The tree is not the problem. How
many elements does each thread load, and are those loads coalesced?

**Slower than `cudaMemcpy` of the same data.** You are reading the input more
than once. Count your global loads.

---

## 8. Exercises

### 7.1 - The bound

(a) Minimum possible time to sum 256M fp32 on your GPU.
(b) What GB/s would the harness report for a perfect implementation?
(c) If your kernel reads the input twice, what is the best achievable time?
(d) The two-kernel approach writes `gridDim` partials and reads them back. For
    `gridDim = 4096`, what fraction of total traffic is that? Does it matter?

### 7.2 - Trace the trees

Block of 8 threads, `sdata = [1,2,3,4,5,6,7,8]`. For each scheme, tabulate
`sdata` after every step, and for each step record (i) how many threads are
active, (ii) how many warps would be active if this were 256 threads, (iii) how
many shared-memory bank conflicts a full warp would see.

(a) `tid % (2*s) == 0`, s = 1, 2, 4
(b) `index = 2*s*tid`, same s
(c) `tid < s`, s = 4, 2, 1

Now explain, from your own table, why (c) is better than (b) is better than
(a). Do not look it up. The table contains the whole argument.

### 7.3 - The shuffle reduction

(a) Write out all 5 steps of `warp_reduce_sum` for `v = lane_id`, lane by lane.
(b) Which lanes hold the correct total at the end?
(c) Change `__shfl_down_sync` to `__shfl_xor_sync`. Tabulate again. Which lanes
    hold the total now? Does it cost more?
(d) When would you prefer the xor version?
(e) A block of 1024 threads = 32 warps. Design the two-level block reduction.
    How many barriers? How many shared floats? Can you do it with *one*
    barrier?

### 7.4 - Climb the ladder

This is the main exercise. In [`07-reduce`](../exercises/07-reduce) you
implement up to seven versions. For **each** version, before you write it:

```
predicted time  : ______
predicted GB/s  : ______
bottleneck      : tree / memory / divergence / banks / launch
```

and after measuring, fill in:

```
measured        : ______
speedup over previous : ______x
what the previous version was wasting : ______
```

If a version does not give the speedup you predicted, find out why before
continuing. The written table is the deliverable; the kernels are just how you
produce it.

### 7.5 - The cross-block decision

Implement the final combine all three ways: two kernels, atomics, and
last-block-finishes (an `atomicAdd` on a counter, with the final block doing
the last pass).

(a) Measure all three at n = 256M.
(b) Measure all three at n = 4096. What changed, and why?
(c) Run each 100 times and check bit-exact reproducibility. Which are
    deterministic?
(d) When is each the right choice? Write the decision rule you would give a
    colleague.

### 7.6 - Generalise

Extend your best reduction to:

(a) max
(b) argmax (returning the index; define your tie-break)
(c) sum and sum-of-squares in one pass
(d) the pair `(max, sum of exp(x - max))` with the online-softmax combine rule

For (d), write down the combination rule for two partial results
`(m1, s1)` and `(m2, s2)`, and *prove it is associative*. You will use this
exact operator in Chapter 10 and again in Chapter 19. If you can derive it
here, FlashAttention will be a tiling exercise rather than a mystery.

---

## Exercises to run

- [`07-reduce`](../exercises/07-reduce) - the ladder. Bar: 85% of copy ceiling.

---

## What you should now believe

- A reduction is memory-bound; copy bandwidth is the target.
- There is no cheap global sync, and that constraint dictates the structure.
- Sequential addressing beats interleaved, on both divergence and banks.
- Shuffles beat shared memory for the last 32 elements, and for the whole warp
  level.
- The decisive optimisation is elements per thread, not tree shape.
- Parallel summation is *more* accurate than sequential.

Next: [Chapter 08 - Scan](08-scan.md), where the operator is no longer
symmetric and everything gets harder.
