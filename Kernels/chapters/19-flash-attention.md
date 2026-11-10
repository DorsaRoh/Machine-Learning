# Chapter 19 - Fused attention to FlashAttention

> FlashAttention is usually taught as a clever trick someone invented. It is
> not. It is the only thing you can do once you write down attention's
> arithmetic intensity, and you already have every tool needed to derive it:
> the roofline (Chapter 01), tiling (Chapter 06), online softmax (Chapter 10),
> register-tiled GEMM (Chapter 12) and tensor cores (Chapter 13). This chapter
> is where all of them meet.

---

## 1. Attention, as written in the paper

For one head, with sequence length `N` and head dimension `d`:

```
S = Q K^T / sqrt(d)          [N, N]
P = softmax(S, dim=-1)       [N, N]
O = P V                      [N, d]
```

Two matmuls with a softmax between them. The matmuls are fine. The softmax is
the entire problem, and not for the reason people expect -- not numerics, but
*the fact that it needs a whole row before it can emit anything*.

---

## 2. Do the roofline arithmetic before anything else

Take `N = 4096`, `d = 64`, 32 heads, fp16 in, fp32 accumulate.

**FLOPs**, per head:

```
QK^T : 2 N^2 d = 2 * 4096^2 * 64 = 2.15 GFLOP
PV   : 2 N^2 d                   = 2.15 GFLOP
total per head  = 4.3 GFLOP
32 heads        = 137 GFLOP
```

**HBM traffic** for the three-kernel version, per head. `S` and `P` are
`[4096, 4096]` fp16 = 33.5 MB each:

```
kernel 1 (QK^T) : read Q, K (1.0 MB)        write S (33.5 MB)
kernel 2 (softmax): read S (33.5 MB)        write P (33.5 MB)
kernel 3 (PV)   : read P, V (34.0 MB)       write O (0.5 MB)
                                            ----------------
                                  per head:  ~136 MB
                                  32 heads:  ~4.35 GB
```

**Arithmetic intensity** = 137 GFLOP / 4.35 GB = **31 FLOP/byte**.

On an A100 (312 TFLOP/s fp16 tensor, 1.55 TB/s), the ridge point is 201
FLOP/byte. You are at 31. You are not 15% memory-bound; you are 6.5x on the
wrong side of the knee.

```
time if compute-bound : 137 GFLOP / 312 TFLOP/s  = 0.44 ms
time if memory-bound  : 4.35 GB  / 1.55 TB/s     = 2.81 ms
```

The hardware can do the math in 0.44 ms and spends 2.81 ms moving an
intermediate it never needed. **Six sevenths of the time is pure waste**, and
all of it comes from one decision: materializing `S` and `P`.

There is a second cost, and in production it is the one that bites first:
`[N, N]` per head is `O(N^2)` *memory*. At N = 16384 and 32 heads, `S` alone is
16 GB. The kernel does not get slow, it gets an out-of-memory error. This is
why context lengths were stuck where they were.

---

## 3. Why you cannot just fuse it

You know what to do with a memory-bound chain of kernels: fuse it (Chapter 10).
Compute a tile of `S`, softmax it, multiply by `V`, never write `S` to HBM.

Except the middle step is wrong. `softmax(S[i, :])` needs

```
m_i = max over the WHOLE row
l_i = sum of exp(S[i, j] - m_i) over the WHOLE row
```

and the whole row is `N` wide. You cannot normalize a tile of `S` until you
have seen the tiles to its right, which you have not computed yet.

This is exactly the obstacle from Chapter 10 section 4, one dimension up. And
the resolution is the same one.

---

## 4. Online softmax, restated as a merge

From Chapter 10: the pair `(m, l)` -- running max and running sum of
`exp(x - m)` -- forms a mergeable state. Given block `j` with local max `m_j`
and local sum `l_j`:

```
m_new = max(m, m_j)
l_new = l * exp(m - m_new) + l_j * exp(m_j - m_new)
```

Associative, so you can fold blocks in any order, streaming.

Attention needs one more thing: it is not just accumulating a scalar sum, it is
accumulating `O = sum_j P_j V_j`, a `[Br, d]` matrix. The partial `O` was
computed with a stale max, so it must be rescaled by the same correction
factor:

```
m_new = max(m, rowmax(S_j))
corr  = exp(m - m_new)                 # one scalar per row
P_j   = exp(S_j - m_new)               # [Br, Bc]
l     = l * corr + rowsum(P_j)         # [Br]
O     = O * corr + P_j @ V_j           # [Br, d]   <- the rescale, in registers
m     = m_new
```

and at the very end, once:

```
O = O / l
```

That is the whole algorithm. `O` is kept **unnormalized** through the loop,
carrying the correction forward, and divided exactly once. Everything else in
FlashAttention is engineering around this recurrence.

---

## 5. The algorithm

```
for each block of Br query rows (parallel: one threadblock each):
    load Q_i  [Br, d] into shared/registers    # once
    O = 0 [Br, d] in registers
    l = 0 [Br]  , m = -inf [Br]

    for each block of Bc key/value rows:        # sequential, inside the block
        load K_j, V_j  [Bc, d]  into shared
        S_j = Q_i @ K_j^T * scale        [Br, Bc]   <- tensor cores
        (apply mask)
        m_new = max(m, rowmax(S_j))
        P_j   = exp(S_j - m_new)
        corr  = exp(m - m_new)
        l     = l * corr + rowsum(P_j)
        O     = O * corr + P_j @ V_j                <- tensor cores
        m     = m_new

    O = O / l
    write O_i                                       # once
    write L_i = m + log(l)                          # for the backward pass
```

Note the shape of it: **two chained matmuls where the second one's A operand
never leaves registers.** `S_j` is produced by the first `mma` and consumed by
the second without touching HBM and, in a good implementation, without touching
shared memory either.

### The new traffic

Per head, with `Br = Bc = 128`:

```
Q read once         : 0.5 MB
O written once      : 0.5 MB
L written           : negligible
K, V read once per Q block = N/Br = 32 times : 32 * 1.0 MB = 32 MB
                                               ---------------
                                       total:  ~33 MB  (was 136 MB)
```

4x less, and `O(N)` memory instead of `O(N^2)`.

Better than it looks: the `N/Br` re-reads of `K` and `V` are made by *different
threadblocks running concurrently*, and `K + V` for one head is 1 MB against
40 MB of L2. Most of those re-reads never reach HBM at all. Measure the real
number with `dram__bytes.sum` rather than trusting either estimate.

The general statement, worth memorizing: tiling turns traffic of `O(N^2 d)`
into `O(N^2 d^2 / M)` where `M` is the SRAM you tile into. Bigger tiles,
strictly less traffic -- until registers run out.

### What sets the tile size

`O` lives in registers for the whole loop: `Br * d` floats per block. At
`Br = 128`, `d = 64`, 128 threads, that is `128 * 64 / 128 = 64` registers per
thread **for the accumulator alone**, before `S_j`, addresses, or `m` and `l`.
The budget is 255.

This is the single tightest constraint in the kernel, and it is why `d` larger
than 128 needs a different structure. Work out your own budget in exercise 19.1
before you write a line.

---

## 6. FlashAttention-2: the same math, different schedule

FA1 was ~25-40% of peak. FA2 is ~60-70%. The algorithm did not change. Three
scheduling decisions did.

**Loop order.** FA1 looped over `K` blocks outside and `Q` blocks inside,
writing partial `O`, `m`, `l` back out and reloading them. FA2 puts `Q`
outside, so `O`, `m`, `l` stay in registers for the block's whole life and are
written exactly once.

**Fewer non-matmul FLOPs.** Non-matmul work runs on the CUDA cores at roughly
1/16 of tensor-core throughput, so a FLOP of `exp` costs what 16 FLOPs of
matmul cost. FA2 removes the per-iteration division (defer it), and uses
`exp2f` with the scale folded into `Q` (`exp(x) = exp2(x * log2(e))`) so the
hot path is a single `MUFU.EX2`. Count these in exercise 19.2; the result is
surprising.

**Warp partitioning.** FA1 split `K` across the four warps of a block, so each
warp held a partial `O` for all rows and the warps had to reduce through shared
memory every iteration. FA2 splits `Q` across warps: each warp owns 32 rows
entirely, needs all of `K` and `V` (which are in shared memory anyway), and
**never communicates**. Same math, no cross-warp reduction, no barrier.

The lesson generalizes past attention: when you partition work across warps,
partition along the axis that makes each warp's output private.

---

## 7. The details that decide whether yours is fast

**Causal masking.** For a causal model, blocks entirely above the diagonal
contribute nothing -- skip them, do not mask them. That halves the work, and
creates a load-imbalance problem: block 0 does 1 iteration, block `N/Br - 1`
does `N/Br`. Fix it in the block-to-row mapping, not with a mask. Only the
diagonal blocks need an element-wise mask.

**Masking by `-inf` vs by a large negative.** `exp(-inf - m)` is 0, which is
what you want, but a fully masked row gives `l = 0` and `O / 0 = NaN`. Decide
deliberately what a fully masked row should produce.

**The `S_j` layout problem.** The output fragment layout of the first `mma` is
not the input fragment layout the second `mma` wants for its A operand. You
either shuffle through shared memory, or pick `mma` shapes so the layouts line
up. This is the fiddliest part of writing one by hand, and it is where
`ldmatrix` and the fragment maps from Chapter 13 earn their keep.

**The rowmax and rowsum.** `S_j` is spread across a warp's registers in the
`mma` fragment layout, so `rowmax` is a reduction across specific *lanes*, not
across threads in index order. Work out which lanes hold which elements of a
row for your chosen `mma` shape (Chapter 13 section 4), then it is a
`__shfl_xor_sync` butterfly over just those lanes. Do not route it through
shared memory.

**The rescale is `Br * d` multiplies per iteration.** At `Br = 128, d = 64`
that is 8192 non-matmul FLOPs per iteration, against 2 `[128,128]x[128,64]`
matmuls. Check whether it is free before you optimize it, and check again after
you make everything else fast.

**Dropout and softcap** both go in the same place and cost the same way:
element-wise on `S_j` or `P_j` while it is still in registers. Dropout needs a
counter-based RNG (`curand` Philox) so the backward pass can regenerate the
same mask from the seed and offset rather than storing it -- storing it would
reintroduce the `O(N^2)` you just eliminated.

**Decode is a different kernel.** With one query row (`Br = 1`), there is no
matmul: it is a GEMV against the KV cache, and it is purely memory-bound on the
cache. The tiling in this chapter buys nothing. FlashDecoding splits the
*sequence* across blocks and combines the partial `(O, m, l)` with exactly the
merge from section 4 -- which works because that merge is associative. Chapter
21 picks this up.

---

## 8. Common failure modes

**Wrong by a factor that varies per row.** Something in the rescale chain.
Check: is `O` rescaled by `corr` *before* `P_j @ V_j` is added, and is `l`
rescaled by the same `corr` in the same iteration?

**NaNs that appear only with masking.** A row with no unmasked elements.

**NaNs that appear only with long sequences.** The running max is not actually
tracking the max, or `exp` is being applied before the subtraction.

**Correct, but no faster than three kernels.** Look at `dram__bytes.sum`. If it
is close to the `O(N^2)` figure, something is spilling: either `O` to local
memory (check for `LDL`/`STL` in SASS) or `S_j` to shared and back.

**Fast at N = 1024, slow at N = 8192.** Check occupancy and spills at the
larger size, and check whether your grid is large enough at the smaller one.
These are different problems; measure both.

**Causal version is less than 2x the non-causal version.** Load imbalance.
Look at the per-block timings, not the aggregate.

**Numerically fine in fp16 until it is not.** `P_j @ V_j` must accumulate in
fp32. `exp` must be computed in fp32. Only the `mma` inputs are fp16.

---

## 9. Exercises

### 19.1 - Budget the registers before you write anything

For `Br = 128`, `Bc = 128`, `d = 64`, 128 threads per block, fp32 accumulators:

(a) Registers per thread for `O`.
(b) Registers per thread for one `S_j` fragment set.
(c) `m`, `l`, and the correction factors.
(d) Addresses, loop counters, and the `K`/`V` staging if you use `cp.async`.
(e) Total. Against 255, what is left?
(f) Now redo it for `d = 128`. What breaks, and what is the smallest change to
    the tiling that fixes it?
(g) Shared memory for `K_j` and `V_j` at these sizes. Blocks per SM? Is that
    enough occupancy, and does this kernel actually need much?

### 19.2 - Count the non-matmul FLOPs

Per iteration of the inner loop, at `Br = Bc = 128`, `d = 64`:

(a) Matmul FLOPs.
(b) `exp` operations.
(c) Rescale multiplies (`O * corr`).
(d) Reduction operations for `rowmax` and `rowsum`.
(e) Using 312 TFLOP/s tensor and 19.5 TFLOP/s fp32 CUDA-core throughput,
    convert each to time.
(f) What fraction of the iteration is non-matmul? Is that where you expected?
(g) Which single line would you attack first, and what is the maximum speedup
    available from attacking it? (Amdahl, before you optimize.)

### 19.3 - Derive the traffic formula

(a) For tile sizes `Br`, `Bc`, write total HBM traffic as a function of `N`,
    `d`, `Br`, `Bc`.
(b) Minimize it. Which tile dimension matters, and why only one?
(c) Substitute the SRAM constraint and recover the `O(N^2 d^2 / M)` form.
(d) At `N = 4096`, `d = 64`: traffic for `Br = 32`, `64`, `128`, `256`.
(e) At which point does the register budget stop you?
(f) Predict the fraction of `K`/`V` re-reads that L2 absorbs on your GPU, then
    measure `dram__bytes.sum` and compare.

### 19.4 - Prove the merge is associative

(a) Write the state `(m, l, O_unnormalized)` and the merge operator for two
    states.
(b) Prove associativity.
(c) Prove that folding left-to-right gives the mathematically exact softmax
    result, in exact arithmetic.
(d) In floating point it is not exact. Bound the error, or construct an input
    where the order matters measurably.
(e) Why does associativity matter *practically* here? (Two answers: one about
    FlashDecoding, one about the backward pass.)

### 19.5 - The three-kernel baseline, honestly

Before the fused version, build the slow one -- you need it as a correctness
oracle and as the number you are beating:

(a) Implement `QK^T`, softmax, `PV` as three kernels using cuBLAS for the
    matmuls and your Chapter 10 softmax. (The exercise harness ships one; write
    yours first, then read theirs and find the differences.)
(b) Measure. Compute achieved bandwidth and achieved TFLOP/s.
(c) Where is it on the roofline? Does it match your section 2 prediction? If
    not, find the discrepancy before continuing -- a wrong model will mislead
    you for the rest of the chapter.
(d) Find the largest `N` it can run before OOM. Plot memory against `N` and
    confirm the quadratic.

### 19.6 - Read the three papers in order

FlashAttention, FlashAttention-2, FlashAttention-3.

(a) For each, what *exactly* changed from the previous one?
(b) Which changes are algorithmic and which are scheduling?
(c) FA2 reports ~2x over FA1 with identical math. Attribute that 2x to the
    three changes in section 6, with numbers.
(d) FA3 uses warp specialization and TMA (Chapter 14). Why do those only pay
    off on Hopper?
(e) What is the next bottleneck after FA3, and what hardware change would be
    needed to remove it?

---

## Exercises to run

- [`19-flash-fwd`](../exercises/19-flash-fwd) - fused attention forward.
  Bar: `O` and `L` within 2e-3, `O(N)` workspace, >= 2.5x the three-kernel
  cuBLAS baseline. (Then compare against PyTorch SDPA yourself; 40% of it by
  hand is a good result.)

---

## What you should now believe

- Attention is memory-bound by a factor of roughly 6, and the cause is one
  intermediate that never needed to exist.
- Softmax blocks fusion only if you insist on computing it in one pass; the
  `(m, l)` state makes it streamable, and the price is a rescale of the
  accumulator.
- FlashAttention is tiling (Chapter 06) plus online softmax (Chapter 10) plus
  register-resident accumulation (Chapter 12). You could have derived it.
- The `O(N)` memory is at least as important as the speedup. It is why long
  context exists.
- FA1 to FA2 is a scheduling story: loop order, non-matmul FLOP count, and
  partitioning warps along the axis that makes their output private.
- Non-matmul work on CUDA cores costs ~16x per FLOP what tensor-core work does.
  Count it.

Next: [Chapter 20 - Backward passes](20-backward-passes.md). Everything you
just built, run in reverse, where the recompute-versus-store trade reappears
with different numbers.
