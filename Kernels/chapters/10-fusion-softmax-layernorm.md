# Chapter 10 - Fusion: softmax and LayerNorm

> The most valuable chapter for anyone doing machine learning. Nearly every
> non-GEMM kernel in a transformer is memory-bound, which means the only lever
> is passes over data, which means **fusion**. And the sharpest version of that
> idea -- online softmax -- is the mathematical core of FlashAttention.

---

## 1. Why fusion is the whole game

A sequence of elementwise operations, done separately:

```
y = relu(x)      read N, write N
z = y * s        read N, write N
w = z + b        read N, write N
                 total: 6N bytes
```

Fused into one kernel:

```
w = relu(x) * s + b    read N, write N
                       total: 2N bytes -> 3x faster
```

The math is identical. The arithmetic was always free -- you are at 1% of the
compute roof either way. **All you did was stop writing intermediates to HBM
and reading them back.**

This is why `torch.compile`, XLA, TensorRT and every other ML compiler exist:
their primary job is fusing elementwise chains. It is also why hand-written
fused kernels for attention, normalization and activations are worth writing:
the compiler fuses what it can prove safe, and a human can fuse more.

**The rule:** for memory-bound work, count passes over the data. That count
*is* your runtime.

---

## 2. Softmax

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

### Numerical stability first

`exp(x)` overflows fp32 at x > 88 and fp16 at x > 11. Attention scores reach
into the tens routinely. So nobody computes the naive formula; everyone
subtracts the max:

```
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m),   m = max_j x_j
```

Mathematically identical (the factor `exp(-m)` cancels), numerically safe: the
largest exponent is now exactly 0, so `exp` returns at most 1 and the
denominator is at least 1. **This is not optional, and it is the reason the
algorithm needs a max pass at all.**

### Counting passes

The straightforward implementation:

```
pass 1: m = max(x)
pass 2: s = sum(exp(x - m))
pass 3: y = exp(x - m) / s
```

Three reads, one write = **4N bytes of traffic**. If the row fits in registers
or shared memory, passes 2 and 3 reuse the cached copy and you get down to
**2N** -- one read, one write, which is optimal.

So for rows that fit on-chip, the answer is easy: load once, keep it, done. The
interesting case is rows too big to hold, and *that* is where online softmax
earns its place -- and where attention lives, because an attention row is the
whole key sequence.

### Online softmax: one pass

Compute the max and the sum **simultaneously**, in a single pass, by fixing up
the running sum whenever the max changes.

Keep `(m, s)`: the max so far and the sum of `exp(x - m)` so far. On seeing a
new value `x`:

```
m_new = max(m, x)
s_new = s * exp(m - m_new) + exp(x - m_new)
```

The factor `exp(m - m_new)` **rescales** the old sum into the new max's frame.
If the max did not change, it is `exp(0) = 1` and costs nothing meaningful.

Merging two partial results `(m1,s1)` and `(m2,s2)` -- the form you need for a
parallel reduction:

```
m = max(m1, m2)
s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

**This operator is associative.** That is the entire reason the trick works: it
means you can reduce with it in a tree, across lanes with shuffles, across
warps with shared memory, across blocks, and across tiles of an attention
matrix in any order. Verify associativity yourself (exercise 10.3); do not take
it on faith.

Traffic: one read for the combined max/sum pass, one read to normalise, one
write = **3N**, or **2N** if the row is cached after the first pass.

The payoff is not really the single-row speedup. It is that **you never need to
see the whole row at once**. You can process a row in tiles, carrying `(m, s)`
forward, and the answer is exactly right. Hold that thought until Chapter 19;
it is the entire idea of FlashAttention.

---

## 3. LayerNorm

```
mean = (1/D) sum_i x_i
var  = (1/D) sum_i (x_i - mean)^2
y_i  = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
```

### One pass via the sum-of-squares identity

```
var = E[x^2] - E[x]^2
```

So accumulate `sum(x)` and `sum(x*x)` together in one pass, then compute both
statistics at the end. Two accumulators, one read.

**The catch:** `E[x^2] - E[x]^2` is the textbook example of catastrophic
cancellation. If the mean is large relative to the standard deviation, the two
terms are nearly equal and you subtract away all your significant digits. With
`x ~ N(1000, 1)` in fp32 you can lose the variance entirely and even get a
*negative* result, whose square root is NaN.

In practice for LayerNorm over activations this is usually acceptable (they are
roughly zero-mean, and you accumulate in fp32), which is why many production
kernels do use it. But "usually acceptable" is a choice you should make
knowingly.

### Welford's algorithm

The numerically stable one-pass alternative:

```
count += 1
delta  = x - mean
mean  += delta / count
delta2 = x - mean            (note: the UPDATED mean)
M2    += delta * delta2
var    = M2 / count
```

And the parallel merge, which is what you need on a GPU:

```
merge (n_a, mean_a, M2_a) with (n_b, mean_b, M2_b):
  n     = n_a + n_b
  delta = mean_b - mean_a
  mean  = mean_a + delta * n_b / n
  M2    = M2_a + M2_b + delta^2 * n_a * n_b / n
```

Associative again, so it reduces in a tree just like softmax's operator. Costs
a division per element and a few more flops in the merge -- free, since you are
memory-bound. Choose Welford unless you have measured that you do not need it.

### RMSNorm

```
y_i = gamma_i * x_i / sqrt(mean(x^2) + eps)
```

No mean subtraction. One accumulator, no cancellation risk, cheaper. This is
why Llama and most modern LLMs use it, and it is a good example of an
architecture choice driven partly by kernel considerations.

---

## 4. Choosing the parallel decomposition

For a `[rows, cols]` tensor with a per-row reduction, the layout decision is
the whole design:

**One block per row** (the usual choice). The row's reduction happens with
shuffles and shared memory inside the block; no cross-block communication.
Reads are coalesced if the row is contiguous. Needs `cols` to be big enough to
fill a block with work -- for 1024 columns and 256 threads, each thread handles
4 elements. Good for `cols` from a few hundred to a few tens of thousands.

**One warp per row.** Better when rows are short (say 32-256), because a block
per row would leave most of the block idle. Several rows per block, each warp
independent, zero barriers.

**One thread per row.** Each thread loops over its row. Terrible: consecutive
threads are `cols` elements apart, so every access is maximally strided. This
is the trap from Chapter 03 exercise 3.3(d), and it is the most common
beginner's softmax.

**Multiple blocks per row.** Necessary when a row is enormous (vocabulary-sized
softmax over 128k logits). Requires cross-block combination -- and because the
softmax operator is associative, you can do it with the same two-kernel or
atomic strategies as Chapter 07.

---

## 5. What to fuse in a transformer

Real wins, roughly in order of value:

- **Attention**: QK^T, scale, mask, softmax, PV. Unfused materialises an
  `N x N` score matrix twice. Fused never materialises it. This is
  FlashAttention, and it changes memory from `O(N^2)` to `O(N)`.
- **LayerNorm + residual add**: the residual is read anyway, so folding the add
  in is free.
- **Bias + activation**: `gelu(x + b)` as one kernel.
- **Dropout + residual + LayerNorm**: three passes into one.
- **The MLP block**: `gelu(x @ W1 + b1)` fuses the epilogue into the GEMM.
- **Softmax + cross-entropy**: the loss needs the log-sum-exp the softmax
  already computed. Never compute softmax then take its log.
- **Optimizer steps**: Adam touches 4 tensors per parameter. Fusing the whole
  update, and fusing across *parameters* (a "foreach"/multi-tensor kernel), is
  one of the biggest easy wins in a training loop.

A useful habit: open a profiler timeline of a training step and look for
adjacent small kernels. Every pair of them is a fusion candidate, and the
savings are exactly the intermediate's traffic.

---

## 6. Common failure modes

**NaN in softmax.** No max subtraction. Or a fully-masked row, where every
entry is `-inf`, so the max is `-inf` and you compute `exp(-inf - -inf)`. Both
are real and the second one bites in causal attention with padding.

**Softmax is slow despite a perfect reduction.** Count the passes. Did you read
the row three times?

**LayerNorm variance is negative.** `E[x^2] - E[x]^2` cancellation. Welford.

**Results differ from PyTorch in the 4th decimal.** Accumulation order, or
accumulating in fp16 rather than fp32. Chapter 15.

**One thread per row.** Check your sectors per request. If it is 32, this is
your bug.

**Fused kernel slower than the unfused pair.** Register pressure or occupancy
dropped, or you now read an input twice that you previously read once. Count
the bytes again, carefully.

**Short rows are slow.** A block per row with 32 useful elements wastes the
block. Switch to a warp per row.

---

## 7. Exercises

### 10.1 - Count the passes

For `[8192, 4096]` fp32 on your GPU:

(a) Minimum possible traffic for softmax. Minimum time.
(b) Traffic and time for the three-pass version with no caching.
(c) Traffic for the two-pass version with the row held in registers.
(d) Traffic for online softmax, assuming the row does *not* fit on-chip.
(e) Which wins, and under what condition does the answer flip?

### 10.2 - Derive the online update

Starting from `(m, s)` meaning "max so far, and sum of exp(x_i - m) so far":

(a) Derive the single-element update rule from scratch.
(b) Derive the two-partial merge rule.
(c) Show both reduce to the standard formula when applied to a whole row.
(d) What is the identity element?
(e) How many `exp` calls per element does a naive implementation of the update
    make, and can you get it down to one?

### 10.3 - Prove associativity

(a) Prove the online softmax merge is associative. Do it symbolically, all
    cases.
(b) Prove the Welford merge is associative.
(c) Is either *commutative*? Does commutativity matter for a tree reduction on
    a GPU? For a `__shfl_xor_sync` butterfly?
(d) In floating point, are these exactly associative? If not, what does that
    imply for reproducibility, and does it change which of the two
    mathematically-equivalent orderings you should prefer?

### 10.4 - Cancellation, concretely

(a) For `x ~ N(1000, 1)` with D = 4096, compute `E[x^2]` and `E[x]^2` in fp32.
    How many significant digits survive the subtraction?
(b) Construct a small input where it produces a negative variance.
(c) Run the same input through Welford. Compare to a float64 reference.
(d) For `x ~ N(0, 1)`, how bad is the naive version? Is it bad enough to care?
(e) Production LayerNorm kernels often use the naive identity with fp32
    accumulation. Under what assumption about the input is that safe, and when
    does that assumption fail in a real network?

### 10.5 - Design the decomposition

For each shape, choose a decomposition and justify it with numbers (threads per
row, elements per thread, bytes per warp access, barriers):

(a) `[1e6, 32]` -- many tiny rows
(b) `[8192, 1024]` -- the typical transformer case
(c) `[128, 128000]` -- vocabulary softmax
(d) `[1, 1e9]` -- one enormous row

For (c) and (d), cross-block combination is required. Write the protocol.

### 10.6 - Fusion audit

Take a transformer block: LayerNorm, QKV projection, attention, output
projection, residual, LayerNorm, MLP up, GELU, MLP down, residual.

(a) List every intermediate tensor and its size for batch 8, sequence 2048,
    hidden 4096.
(b) Total HBM traffic if nothing is fused.
(c) Total if you fuse every elementwise chain.
(d) Total if you additionally fuse attention (never materialise the scores).
(e) Overall speedup for the memory-bound portion.
(f) Which single fusion is worth the most? Is it the one you expected?

This is the calculation that justifies FlashAttention's existence. Do it
carefully; Chapter 19 begins where this ends.

---

## Exercises to run

- [`10-softmax`](../exercises/10-softmax) - bar: 85% of copy ceiling.
- [`10-layernorm`](../exercises/10-layernorm) - bar: 85%, with a numerical
  stability test you must also pass.

---

## What you should now believe

- For memory-bound work, passes over data *are* the runtime, and fusion is how
  you reduce them.
- Softmax must subtract the max; this is a correctness requirement.
- Online softmax computes max and sum in one pass, and its merge operator is
  associative -- which is what lets attention be tiled.
- `E[x^2] - E[x]^2` is fast and dangerous; Welford is stable and effectively
  free on a memory-bound kernel.
- The row-to-thread mapping decision determines coalescing and therefore
  performance.

Next: [Chapter 11 - GEMM I](11-gemm-i.md). Part II is done; you now have every
tool needed to build a fast matrix multiply.
