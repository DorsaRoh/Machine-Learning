# Chapter 22 - Sparsity, MoE and ragged batching

> A GPU is a machine for applying the same operation to a rectangle. Real
> workloads are not rectangles: sequences have different lengths, experts
> receive different numbers of tokens, and most of the entries in a sparse
> matrix are not there. Every technique in this chapter is a way of
> reconstructing a rectangle, or of paying as little as possible for not
> having one.

---

## 1. The cost of pretending

The default answer to ragged data is padding, and padding has a precise price.

A batch of 32 sequences with lengths drawn from a realistic distribution --
mean 512, max 2048:

```
padded work    : 32 * 2048            = 65536 token-slots
useful work    : 32 * 512             = 16384 token-slots
waste          : 75%
```

For attention it is worse, because the cost is quadratic in the padded length:

```
padded : 32 * 2048^2 = 134e6
useful : sum of L_i^2, roughly 32 * 512^2 (more, since E[L^2] > E[L]^2) ~ 8.4e6
waste  : ~94%
```

You are buying a GPU and using 6% of it. That is the motivation for every
"varlen" kernel in every serving stack.

The general shape: **padding waste grows with the variance of the length
distribution and with the exponent of the operation.** Linear layers waste
`1 - mean/max`; attention wastes far more. Which also tells you where to spend
effort first.

---

## 2. Ragged layouts

Store sequences concatenated, with an offsets array:

```
tokens    : [ s0 s0 s0 | s1 s1 | s2 s2 s2 s2 s2 ]     total_tokens rows
cu_seqlens: [ 0, 3, 5, 10 ]                            batch+1 entries
```

`cu_seqlens` is a prefix sum -- Chapter 08, in production. Sequence `i` owns
rows `cu_seqlens[i]` to `cu_seqlens[i+1]`.

For element-wise and linear layers this changes nothing: `[total_tokens, H]` is
a perfectly good rectangle and the GEMM does not care where sequence
boundaries are. **This is the key observation**: most of a transformer is
indifferent to raggedness. Only attention is not, because only attention mixes
across the sequence dimension.

For attention, the launch itself becomes data-dependent:

```
blocks for sequence i = ceil(len_i / Br)
grid = sum over i of that
```

Two ways to build that grid, and the choice recurs everywhere in this chapter:

1. **Host-side.** Prefix-sum the block counts on the host (or in a tiny kernel),
   launch exactly that many blocks, and have each block binary-search the
   offsets array to find which sequence and which tile it owns. Costs a
   `cudaMemcpy` of one integer -- a synchronization point, which inside a decode
   loop you will feel.
2. **Over-launch and exit.** Launch `batch * ceil(max_len / Br)` blocks and have
   the out-of-range ones return immediately. No sync, no host round trip, and
   the empty blocks cost only their launch. Usually the right answer, and the
   reason is Chapter 17: avoiding a host sync is worth more than avoiding
   thousands of no-op blocks.

Watch for the second-order effect: with ragged lengths, blocks finish at wildly
different times, so the tail of the grid is a load-balancing problem. Sorting
sequences by length before batching helps enormously and costs nothing. Causal
masking creates the same imbalance within a single sequence (Chapter 19
section 7).

---

## 3. Gather, scatter, and what they cost

```
gather  : out[i]      = in[idx[i]]
scatter : out[idx[i]] = in[i]
```

Both break the assumption every bandwidth number in this course rests on:
coalescing (Chapter 03).

A warp gathering 32 random rows issues up to 32 separate memory transactions
instead of a handful. If each row is short, you move 32 cache lines to use 32
floats and achieve ~1/8 of peak bandwidth. If each row is long (a full `H`-wide
hidden vector), the *row* is still contiguous and you lose almost nothing --
assign one warp per row and the gather is free.

**So the question is never "is it a gather" but "how many contiguous bytes per
index".** Work out that number before you panic about an index array. 128 bytes
or more and you are fine; 4 bytes and you have a real problem.

Scatter adds a second problem: collisions. Two threads writing the same
destination need atomics, which brings back Chapter 09's contention and
Chapter 15's determinism question.

**Sorting by index turns both into something better.** Sorted indices make a
gather semi-coalesced and make a scatter-add into a segmented reduction with no
atomics at all. A radix sort of a few thousand integers is cheap relative to
what it saves. This is the single most useful trick in this chapter.

---

## 4. Mixture of Experts

Each token is routed to `k` of `E` experts. 8 experts, top-2, 4096 tokens:

```
assignments        : 4096 * 2 = 8192
average per expert : 1024
actual per expert  : 300 to 3000, depending on the router
```

The computation is `E` independent GEMMs with different, data-dependent row
counts. Three ways to do it:

**Dense masking.** Run every token through every expert, multiply by a 0/1
mask. Trivially correct, trivially parallel, and does `E/k = 4x` the necessary
work. Sometimes still the fastest option at small `E`, because it has zero
irregularity. Do not dismiss it without measuring.

**Permute and group.** The standard approach:

```
1. router produces expert_id per (token, slot)         [T*k]
2. histogram expert_ids -> counts per expert           <- Chapter 09
3. exclusive scan counts -> offsets per expert         <- Chapter 08
4. scatter token indices into permuted order           <- sorted gather
5. gather token rows into the permuted buffer
6. grouped GEMM: E matmuls over contiguous row ranges
7. scatter results back, weighted by router probs and summed over k
```

Steps 2 and 3 are exactly your histogram and scan exercises. Step 6 is a
grouped/batched GEMM where each group has a different `M` -- which is why
CUTLASS has a grouped GEMM API and why `torch.bmm` does not help you.

Note what the permutation buys: after step 5 each expert's tokens are
contiguous, so each GEMM is a plain dense GEMM with full coalescing. **You paid
two gathers to turn an irregular problem into `E` regular ones.** Since each
token row is `H` floats contiguous, those gathers are nearly free (section 3).

**Capacity factor.** Fix a maximum tokens-per-expert at `C = capacity_factor *
T * k / E`, pad experts below it, drop tokens above it. Restores the rectangle
and bounds memory, at the cost of dropping real tokens when the router is
unbalanced. Training stacks often do this; inference stacks usually cannot
afford to.

**Load imbalance is the whole difficulty.** If one expert gets 3000 tokens and
another gets 300, the grid finishes when the biggest one does. Fixes, in
increasing order of complexity: auxiliary load-balancing loss during training
(not a kernel fix, but the most effective one); launching blocks proportional
to each expert's count rather than per-expert; and a persistent kernel with a
global work queue.

**The persistent kernel pattern** generalizes past MoE, so it is worth knowing
in its own right:

```
launch exactly as many blocks as fit on the device (occupancy API)
each block loops:
    my_tile = atomicAdd(&global_work_counter, 1)
    if (my_tile >= total_tiles) return
    do tile
```

Blocks pull work instead of being assigned it, so a block that drew a cheap
tile immediately takes another. Converts a static partitioning problem into a
dynamic one for the cost of one atomic per tile. Stream-K GEMM is this idea
applied to the `K` dimension; the tail-effect fix in Chapter 12 is the same
idea applied to wave quantization.

---

## 5. Sparsity, and why unstructured sparsity loses

**2:4 structured sparsity** (Ampere+): in every group of 4 contiguous weights,
exactly 2 are zero. Store 2 values plus a 2-bit index per group, feed the
sparse tensor core, get 2x throughput. It works because the pattern is
*regular* -- the hardware knows exactly where everything is. The constraint is
severe and the model must be trained or fine-tuned into it.

**Unstructured sparsity** (arbitrary zeros, CSR) at 90% sparsity moves 10% of
the data and does 10% of the FLOPs, so it should be 10x faster. It is usually
*slower* than dense. The reasons are all things you now know how to name:
indices are extra traffic; access is uncoalesced; rows have different lengths
so warps diverge and blocks imbalance; tensor cores cannot be used at all, so
you forfeit a 16x advantage to gain a 10x one.

The crossover where CSR SpMM beats dense GEMM is typically above 99% sparsity.
Compute it for your hardware in exercise 22.4 -- it is a clarifying calculation
and it explains why pruning research so rarely produces speedups.

**Block sparsity** is the compromise: zero out `32x32` or `128x128` blocks
instead of individual elements. Each surviving block is a dense tile, so tensor
cores work, loads coalesce, and the only irregularity is which blocks exist --
one level of indirection, amortized over a whole tile. This is what sparse
attention patterns (longformer, bigbird, sliding window) actually compile to,
and it is why they are expressed in blocks rather than in arbitrary masks.

The principle, stated once: **sparsity pays only when the surviving structure
is still large enough to be a rectangle.**

---

## 6. Common failure modes

**Correct output, terrible bandwidth on a gather.** Measure contiguous bytes
per index. If it is small, the fix is the layout or a sort, not the kernel.

**MoE kernel slower than the dense-masked baseline.** At small `E` this can be
correct. Measure the baseline; if the permutation overhead exceeds the `E/k`
saving, you have your answer and it is not a bug.

**One expert dominating the profile.** Load imbalance. Look at per-block
duration, not the total.

**Varlen attention correct for uniform lengths, wrong for ragged.** Off-by-one
in the offsets, or a tile that straddles a sequence boundary. Test with lengths
that are deliberately not multiples of the tile size, and with a length of 1.

**Non-deterministic MoE output.** The combine step uses `atomicAdd` over the
`k` experts. Expected; decide whether you can accept it.

**Scatter-add dominating the time.** Sort the indices and use a segmented
reduction.

**Huge slowdown when the batch composition changes.** You have a host
synchronization in the launch path, and the shape changed. Check for a
`cudaMemcpy` of a count.

---

## 7. Exercises

### 22.1 - Price the padding

With sequence lengths sampled from a log-normal distribution with mean 512 and
the long tail you would see in real traffic:

(a) Padding waste for a linear layer.
(b) Padding waste for attention. Why is it so much worse?
(c) Sort the batch by length and bucket into 4 groups. Recompute both.
(d) At which batch size does length-sorting stop helping?
(e) If you could only implement varlen for one of linear or attention, which,
    and what fraction of the total waste does that recover?

### 22.2 - Build the dispatch, counting everything

4096 tokens, 8 experts, top-2, `H = 4096`:

(a) Bytes moved by the gather in step 5.
(b) Bytes moved by the scatter in step 7.
(c) FLOPs of the expert GEMMs.
(d) FLOPs of the dense-masked alternative.
(e) Which has the better total time on your GPU, and at which `E` does the
    ranking flip?
(f) Now make the router output perfectly imbalanced (all tokens to one expert).
    Redo (e). What does that tell you about worst-case serving latency?

### 22.3 - The work queue

(a) Implement a persistent kernel with an `atomicAdd` work counter over tiles.
(b) How many blocks should you launch? Which API tells you?
(c) Measure the atomic's cost per tile. At what tile granularity does it start
    to matter?
(d) Compare against static assignment for a perfectly balanced workload. Does
    the queue lose? By how much, and is that the price you expected?
(e) Compare for a workload where tile costs vary 10x.
(f) Can a persistent kernel deadlock? Under exactly what conditions? (Chapter
    18.)

### 22.4 - The sparsity crossover

For a `[4096, 4096] x [4096, 4096]` matmul on your GPU:

(a) Dense time, using your measured GEMM throughput.
(b) CSR SpMM time at 50%, 90%, 99%, 99.9% sparsity. Model index traffic,
    uncoalesced access (assume 1/8 efficiency) and no tensor cores.
(c) Where is the crossover?
(d) Redo for block-sparse with `128x128` blocks at the same sparsities.
(e) Where is *that* crossover, and why is it so much lower?
(f) Write the two-sentence version of this result that you would give to
    someone proposing to prune a model for speed.

### 22.5 - Gather granularity

Write a microbenchmark that gathers `n` rows of `w` floats each via a random
index array:

(a) Sweep `w` from 1 to 1024. Plot achieved bandwidth against `w`.
(b) At which `w` do you reach 80% of your copy ceiling?
(c) Sort the index array. Re-measure the whole sweep.
(d) Explain the shape of both curves in terms of sectors and cache lines
    (Chapter 03).
(e) Which `w` corresponds to an MoE token gather? To an embedding lookup? To a
    sparse matrix element? Now you know which of those three to worry about.

---

## Exercises to run

- [`22-moe-dispatch`](../exercises/22-moe-dispatch) - histogram, scan, permute,
  grouped GEMM, combine. Bar: within 2e-2 of the dense-masked reference and
  >= 2.8x faster than it, which is 70% of the ideal `E/k` speedup.

---

## What you should now believe

- Padding waste is computable, and for attention it is usually catastrophic.
- Ragged layouts are a prefix sum plus an index search; most of a transformer
  does not notice raggedness at all.
- A gather costs nothing if each index covers enough contiguous bytes, and
  costs everything if it does not. That number is the whole analysis.
- Sorting indices converts gathers into coalesced reads and scatter-adds into
  segmented reductions. Use it.
- MoE dispatch is histogram plus scan plus permutation plus grouped GEMM --
  every piece of which you have already built.
- Load imbalance, not arithmetic, is the hard part of MoE. Persistent kernels
  with work queues are the general fix.
- Unstructured sparsity loses to dense until absurd sparsity levels. Structure
  is what makes sparsity pay.

Next: [Chapter 23 - Triton](23-triton.md). Part V is done: you can write the
kernels that actually appear in a modern model.
