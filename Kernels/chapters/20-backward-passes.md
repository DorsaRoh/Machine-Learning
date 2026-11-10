# Chapter 20 - Backward passes

> Every fast forward kernel in a training loop creates an obligation: a
> backward kernel that is roughly 2.5x the work, has more operands, has worse
> memory access patterns, and is far harder to debug because the only thing you
> can see is that the loss curve looks slightly wrong. Most of the engineering
> in a real kernel library is here.

---

## 1. The shape of the problem

Forward: `y = f(x)`. Backward: given `dL/dy`, produce `dL/dx = J^T (dL/dy)`,
where `J` is the Jacobian of `f`. You never build `J`; you find the kernel that
applies `J^T`.

There is a mechanical way to get the dataflow right, and it is worth internalizing
because it turns a calculus problem into an indexing problem:

**Transpose the dataflow.**

| Forward | Backward |
|---|---|
| broadcast (one value read by many) | reduce (sum over the many) |
| reduce (many values to one) | broadcast |
| gather `out[i] = in[idx[i]]` | scatter-add `din[idx[i]] += dout[i]` |
| element-wise | element-wise |
| `C = A B` | `dA = dC B^T`, `dB = A^T dC` |
| transpose | transpose |

The entries that cause trouble are the ones that produce *reduce* or
*scatter-add* on the right, because those are where contention, atomics and
non-determinism come from. Read your forward kernel, mark every broadcast and
every gather, and you have a list of the hard parts of the backward before you
write anything.

Three consequences you can count on:

1. **Backward is memory-hungrier.** Where forward read `x` and wrote `y`,
   backward reads `dy`, usually reads `x` or `y` again, and writes `dx`.
2. **Backward has more reductions.** Every parameter broadcast across a batch
   becomes a sum over that batch.
3. **Backward is where atomics appear.** Scatter-adds and parameter gradients
   are the classic sources, and with them come the determinism questions from
   Chapter 15.

---

## 2. Softmax backward

Forward, for one row: `y_i = exp(x_i - m) / sum_k exp(x_k - m)`.

The Jacobian is `dy_i/dx_j = y_i (delta_ij - y_j)`, dense and `[N, N]` -- which
you must never build. Applying `J^T` and simplifying:

```
dx_i = y_i * (dy_i - sum_k dy_k y_k)
```

One row-wide reduction (`sum_k dy_k y_k`), then element-wise. Exactly the same
shape as the forward: a reduction followed by a broadcast. Your Chapter 07
reduction and Chapter 10 fusion apply unchanged, and the same one-pass-versus-
two-pass question appears.

Note that it needs `y`, not `x`. **Saving the output is cheaper than
recomputing from the input**, and this is general: for many element-wise and
normalization ops the backward is expressible in terms of the forward output.
Check that before you decide what to stash.

---

## 3. LayerNorm backward

Forward: `y = g * xhat + b` where `xhat = (x - mean) / sqrt(var + eps)`.

```
dg = sum over batch of (dy * xhat)
db = sum over batch of dy
dxhat = dy * g
dx = (1 / sqrt(var + eps)) * ( dxhat
                             - mean(dxhat)
                             - xhat * mean(dxhat * xhat) )
```

Three things to notice.

**Two row reductions** (`mean(dxhat)` and `mean(dxhat * xhat)`) feed one
element-wise pass. One pass if you keep the row in registers or shared; two
passes if you do not. Same trade as Chapter 10.

**`dg` and `db` reduce over the batch, not the row** -- the orthogonal
direction. For `[B*T, H]` input with `H` on the fast axis, that is a reduction
down columns: coalesced reads, but the partial sums must be combined across
blocks. The standard shape is a two-stage reduction into a `[num_blocks, H]`
scratch buffer followed by a small second kernel. Doing it with `atomicAdd` on
`[H]` instead is one line and usually slower *and* non-deterministic; measure
both, because which wins depends on `H` and on how many blocks you launch.

**Recompute or store `xhat`?** Storing costs a full tensor of traffic. Storing
only `mean` and `rstd` (two floats per row) and recomputing `xhat` costs a
re-read of `x` and a handful of FLOPs. At the sizes in a transformer, the
second one wins -- but it is a bandwidth-versus-FLOPs question with a crossover
you can locate, and exercise 20.2 asks you to locate it.

---

## 4. FlashAttention backward

Given `Q, K, V, O, dO` and the saved log-sum-exp `L` from the forward:

```
recompute: S_ij = Q_i K_j^T * scale
           P_ij = exp(S_ij - L_i)        # L is logsumexp, so P is normalized
dV_j += P_ij^T dO_i
dP_ij = dO_i V_j^T
D_i   = rowsum(dO_i * O_i)               # == rowsum(dP_i * P_i)
dS_ij = P_ij * (dP_ij - D_i)             # softmax backward, from section 2
dQ_i += dS_ij K_j * scale
dK_j += dS_ij^T Q_i * scale
```

Five matmuls against the forward's two, hence ~2.5x the FLOPs.

**`D` is precomputed.** `D_i = rowsum(dO_i * O_i)` needs only row-local data,
so it is a separate tiny kernel over `[N, d]` run before the main loop. That it
equals `rowsum(dP_i * P_i)` -- the quantity softmax backward actually wants --
is the trick that lets the main loop avoid a second pass over `dP`. Verify that
identity yourself (exercise 20.3); it is two lines and it is the reason the
backward fuses at all.

**`L` makes the recompute cheap.** Because the forward stored logsumexp per row
(`O(N)`), `P` is recovered with one `exp`, with no max pass and no second
reduction. This is the payoff for storing `L` rather than `m` and `l`
separately.

### The loop order problem

`dK_j` and `dV_j` accumulate over all `i`, so they want `j` in the outer loop.
`dQ_i` accumulates over all `j`, so it wants `i` outer. You cannot have both,
and the three ways out are a good illustration of a general trade:

1. **`j` outer, `dQ` via `atomicAdd`.** One kernel, but fp32 atomic adds into
   `dQ` make it non-deterministic and add contention.
2. **Two kernels**, one per loop order. Deterministic, clean, and recomputes
   `S` and `P` twice -- paying ~30% more FLOPs to remove the atomics.
3. **`j` outer with a separate fp32 `dQ_accum` buffer** and a split/semaphore
   scheme so each `dQ` block has exactly one writer. Deterministic and single
   pass, at the cost of a `[N, d]` fp32 scratch buffer and real complexity.

The production library offers (1) by default and (3) behind a `deterministic`
flag. Work out for yourself which you would pick for a research codebase versus
for a result someone has to reproduce.

### Recompute versus store, with numbers

Storing `P` from the forward is `N^2` per head -- the thing you deleted in
Chapter 19, back again, and at `N = 8192` it is 134 MB per head. Recomputing it
costs one extra matmul out of five, 25% more FLOPs, on hardware where you are
memory-bound by 6x. Recompute wins by a wide margin, and the margin widens with
`N`.

This is activation checkpointing, in miniature: **trade FLOPs for memory
traffic whenever you are on the memory side of the roofline**, which in
transformer training you almost always are.

---

## 5. Atomics in gradients

Three patterns, in increasing order of pain.

**Parameter gradients reduced over batch** (`dg`, `db`, bias gradients). Fixed
small output, many contributors. Two-stage reduction beats atomics at scale;
atomics are fine when the output is large relative to the number of blocks.

**Embedding gradients.** `dW[token_id] += dout[i]`, where a frequent token
appears thousands of times in a batch. This is the Chapter 09 histogram problem
with 50,000 bins and a power-law distribution. The same tools apply:
privatization, warp aggregation via `__match_any_sync`, or -- usually best --
sort the indices and use segmented reduction so each output row has one writer.

**Scatter-add in sparse and MoE layers.** Chapter 22.

In all three, the determinism question from Chapter 15 is now a question about
*training reproducibility*, not about a unit test. `atomicAdd` on floats means
your run is not reproducible. Decide that deliberately and write it down
somewhere, because somebody will eventually spend a week on it.

---

## 6. Numerics are harder in the backward

Gradients have a wider dynamic range than activations and they skew small. In
fp16 the subnormal threshold is 6e-8 and gradients underflow to zero there --
silently. That is what loss scaling exists for: multiply the loss by `S`,
multiply every gradient by `S` too, divide it out in the optimizer, and halve
`S` when you see a NaN or inf.

bf16 has fp32's exponent range, so underflow is mostly a non-issue, at the cost
of 8 fewer mantissa bits. This is why bf16 training mostly displaced fp16
training, and it is a numerics decision, not a performance one.

Rules that survive contact with reality:

- Accumulate in fp32, always. Store in bf16 if you must.
- Master weights in fp32 even when everything else is bf16.
- The two reductions in LayerNorm backward in fp32, regardless of input dtype.
- A gradient of exactly 0 where you expected something small is underflow until
  proven otherwise.

---

## 7. How to know it is right

Performance is easy to verify and correctness is not, so build the oracle
first.

**Finite differences.** `(f(x + h) - f(x - h)) / (2h)` in float64, `h ~ 1e-5`.
Slow, independent of your implementation, and it catches sign errors, missing
terms and wrong broadcasts. Use it on small shapes.

**`torch.autograd.gradcheck`** in float64, on a small instance. If your op is a
PyTorch custom op (Chapter 24), this is one line and it is the single most
valuable test you can write.

**Compare against the composed reference.** Your fused backward versus
autograd through the unfused forward, same inputs, tight tolerance.

**Check each gradient separately.** `dQ`, `dK`, `dV` fail in different ways:
getting one right tells you nothing about the others, and a transpose error in
`dK` can look like noise in the aggregate.

**Test the edges.** Sequence lengths that are not multiples of the tile. Fully
masked rows. `d` of 32, 64, 128. One query. One key.

---

## 8. Common failure modes

**Gradient correct up to a constant factor.** A missing or doubled `scale`, or
a mean where you wanted a sum.

**`dQ` right, `dK` wrong.** A transpose in the `dS^T Q` matmul.

**Correct for one block, wrong for many.** An accumulation initialized inside
the loop, or a missing atomic.

**Correct in fp32, wrong in fp16.** An accumulator left in fp16, or underflow.
Scale the input up by 1000 and see if the error moves.

**Loss decreases but slower than the reference.** The worst case: a gradient
that is correlated with the true one but wrong -- often a missing term in the
softmax or LayerNorm backward. Numerical gradcheck finds this; watching the
loss does not.

**Non-deterministic across identical runs.** Atomics. Expected; make sure it is
the only source, and that you know it is there.

**Slower than the forward by much more than 2.5x.** Check for a serialized
reduction, atomic contention, or the extra operands blowing your register
budget.

---

## 9. Exercises

### 20.1 - Derive, by hand, on paper

Derive the backward for each, and state in one line what must be saved from the
forward:

(a) `y = x^2`
(b) `y = sum(x)` over a row
(c) `y = max(x)` over a row (what happens at a tie?)
(d) `y = softmax(x)` -- get to the `y_i (dy_i - sum_k dy_k y_k)` form yourself
(e) `y = x / sqrt(sum(x^2) / n + eps)` (RMSNorm)
(f) `C = A B` -- both `dA` and `dB`, with explicit index notation

### 20.2 - The recompute crossover

For LayerNorm backward on `[B*T, H]`:

(a) Bytes moved if you store `xhat` in bf16.
(b) Bytes moved if you store only `mean` and `rstd` and re-read `x`.
(c) Extra FLOPs for (b).
(d) On your GPU's bandwidth and fp32 throughput, times for both.
(e) At which `H` do they cross? Is that value in the range transformers use?
(f) Implement both. Does the measurement match? If not, which assumption in the
    model was wrong?

### 20.3 - The `D` identity

(a) Prove `rowsum(dO_i * O_i) == rowsum(dP_i * P_i)`.
(b) Why does that make a separate preprocessing kernel possible?
(c) What would the main loop have to do without it, and what would it cost?
(d) Write the preprocessing kernel. What is its arithmetic intensity, and what
    is the fastest it could possibly run?

### 20.4 - Choose the loop order

For the flash backward, evaluate all three options from section 4:

(a) Extra FLOPs for each.
(b) Extra memory for each.
(c) Deterministic or not.
(d) At `N = 2048` and `N = 16384`, predict which is fastest.
(e) Implement two of the three and check your prediction.
(f) Which would you ship? Write the two-sentence justification you would put in
    the PR description.

### 20.5 - Build the oracle before the kernel

(a) Write the composed-reference backward in PyTorch (unfused, autograd).
(b) Write a float64 finite-difference check for a small instance.
(c) Confirm the two agree to ~1e-9.
(d) Only now write the kernel.
(e) Write down how long (b) took you, and how many debugging hours it saved.
    This habit is the single highest-leverage one in the course.

### 20.6 - Embedding gradient, three ways

Vocabulary 50,000, batch of 8192 tokens, Zipf-distributed, `H = 1024`:

(a) Naive `atomicAdd`. Measure.
(b) Sort indices, then segmented reduction. Measure.
(c) Warp-aggregated atomics (Chapter 09). Measure.
(d) Which is fastest, and at which skew does the ranking change?
(e) Which are deterministic?
(f) Make the distribution uniform. Does the ranking change? Explain why before
    you run it.

---

## Exercises to run

- [`20-flash-bwd`](../exercises/20-flash-bwd) - the flash backward.
  Bar: `dQ`, `dK`, `dV` within 3e-3, >= 2.0x the unfused cuBLAS backward.
  (Build the float64 gradcheck oracle first anyway.)

---

## What you should now believe

- Backward dataflow is the forward dataflow transposed: broadcasts become
  reductions, gathers become scatter-adds.
- Softmax and LayerNorm backward are reductions plus element-wise passes --
  nothing new, applied to a new expression.
- Flash backward recomputes `S` and `P` from the saved logsumexp because
  recompute is cheap and `O(N^2)` traffic is not.
- `D = rowsum(dO * O)` precomputed in its own kernel is what makes the fusion
  work.
- The loop-order conflict between `dQ` and `dK`/`dV` has three resolutions and
  the choice is about determinism, not speed.
- Backward numerics are harder: fp32 accumulators, and loss scaling if fp16.
- Build the float64 oracle before the kernel. Always.

Next: [Chapter 21 - Quantized kernels](21-quantization.md), where the entire
game is bandwidth and the weights stop being floats.
