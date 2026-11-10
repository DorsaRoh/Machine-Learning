# Chapter 21 - Quantized kernels

> Every other chapter in this course has been about making arithmetic fast.
> This one is about a regime where arithmetic is free and the only thing that
> exists is bytes. Once you see it, a large part of LLM inference engineering
> stops being mysterious.

---

## 1. Decode is a memcpy

Generating one token from a 7B-parameter model at batch size 1 means: read
every weight, multiply it by something, add it up. Every weight is read exactly
once and used in exactly one multiply-add.

```
weights, fp16        : 7e9 * 2 bytes       = 14 GB
FLOPs for one token  : 2 * 7e9             = 14 GFLOP
arithmetic intensity : 14e9 / 14e9         = 1 FLOP/byte
```

One FLOP per byte. The A100 ridge point is 201. You are off by a factor of 200.

```
time per token = 14 GB / 1.55 TB/s = 9.0 ms   ->  111 tokens/s, ceiling
time if compute-bound = 14 GFLOP / 312 TFLOP/s = 0.045 ms
```

The GPU spends 9 ms streaming weights and 0.045 ms doing math. **Your matmul
kernel is a `memcpy` with some arithmetic hidden in its shadow.** Every tensor
core on the die is idle.

So there is exactly one lever: make the weights smaller.

```
fp16  : 14.0 GB  ->  9.0 ms  ->  111 tok/s
int8  :  7.0 GB  ->  4.5 ms  ->  222 tok/s
int4  :  3.5 GB  ->  2.3 ms  ->  435 tok/s
```

That is the entire argument for quantized inference. Not accuracy, not model
size on disk -- tokens per second, bounded by HBM.

It also tells you the only honest metric for a decode kernel: **achieved
bandwidth on the packed weights, as a fraction of your copy ceiling**. TFLOP/s
is meaningless here. If you are at 80% of the ceiling, you are done; no amount
of cleverness in the inner loop will help, and anything that adds bytes will
hurt.

---

## 2. Prefill is a different kernel

Processing a 2000-token prompt, or serving a batch of 64 sequences, reads the
same weights but does 2000x (or 64x) the math. Arithmetic intensity scales with
batch, the problem becomes a GEMM, and you are back in Chapter 12 and 13
territory -- compute-bound, tensor cores, int8 `mma`.

**Two regimes, two kernels, two different sets of right answers.** A kernel
tuned for decode is bad at prefill and vice versa. The crossover is where
arithmetic intensity reaches the ridge point; find it for your GPU in exercise
21.1, because it decides your whole kernel strategy.

---

## 3. What a quantized weight actually is

A quantized tensor is integers plus the metadata to reconstruct approximate
floats:

```
w ~= (q - z) * s
```

`q` is the stored integer, `s` a scale, `z` a zero point. The design space is
how finely `s` and `z` vary.

**Per-tensor.** One scale for the whole matrix. Smallest metadata, worst
accuracy -- a single outlier channel sets the scale for everything.

**Per-channel.** One scale per output row. Metadata is `M` floats, negligible.
Much better, because channel magnitudes in transformers vary by orders of
magnitude.

**Per-group.** One scale per `g` consecutive weights along `K`, with `g` = 64
or 128. This is what int4 schemes use, and the metadata cost is real:

```
7e9 weights, g = 128, fp16 scale + fp16 zero per group
groups = 7e9 / 128 = 54.7e6
metadata = 54.7e6 * 4 bytes = 219 MB   on top of 3.5 GB of packed weights
```

6% more traffic, and 6% of your token rate. Which makes group size a *bandwidth*
decision as much as an accuracy one: `g = 64` doubles that to 12%. Compute the
trade in exercise 21.2 before accepting anyone's default.

**Symmetric vs asymmetric.** Symmetric (`z = 0`, or a fixed midpoint) removes a
subtraction from the inner loop and removes the zero-point storage. Asymmetric
fits skewed distributions better. For int4, asymmetric is usually worth it; for
int8 activations, symmetric usually is.

**Where the schemes differ.** GPTQ chooses `q` to minimize layer output error
using second-order information. AWQ observes that a small fraction of channels
matter disproportionately and scales them up before quantizing. NF4 uses
non-uniform levels matched to a normal distribution. All of them produce the
same *kernel* problem: unpack integers, apply scales, accumulate. The kernel
does not care which solver produced the numbers.

---

## 4. Packing and the layout you choose

Eight int4 values fit in one `uint32`. The obvious layout is sequential:
elements 0-7 in nibbles 0-7. The obvious layout is also the wrong one.

Unpacking happens in the inner loop, once per 8 weights, millions of times per
token. The cost is shifts, masks, and integer-to-float conversions -- and the
conversion is the expensive part, not the masking.

Two things make it cheap, and both come from choosing the bit layout to suit
the *extraction*, not the human reading the file:

**Extract four nibbles at once.** A single `lop3.b32` computes an arbitrary
three-input boolean function in one instruction, and `prmt.b32` permutes bytes.
With the right layout, a `uint32` becomes four half-precision values in a
couple of instructions instead of eight shift-mask-convert sequences.

**Convert without a converter.** An fp16 bit pattern `0x6400 | n` for small `n`
is exactly `1024 + n` as a float. So OR-ing a nibble into a fixed exponent
field *is* the integer-to-float conversion, and the `+1024` folds into the zero
point you were going to subtract anyway. Integer-to-float for free.

These tricks are why production int4 kernels store nibbles in a permuted order
such as `0, 2, 4, 6, 1, 3, 5, 7`: that interleaving is what makes a single
mask-and-OR produce two correctly-paired fp16 values. The packing format exists
to serve the unpacking instruction sequence.

You do not have to invent this. You do have to understand that **the data
layout on disk is part of the kernel**, and that if your unpack is costing more
than a handful of instructions per 8 weights you have a layout problem rather
than an instruction-selection problem.

---

## 5. The dequant GEMV

```
for each output row (or group of rows), one block:
    acc = 0
    for each chunk of K:
        load packed weights          <- the only traffic that matters
        load scale/zero for the group
        load x chunk                 <- small, reused, cache it
        unpack -> fp16
        acc += dequantized * x
    reduce acc across the block
    write y
```

Everything you know applies, with the weights as the only operand that counts:

- **Vectorize the weight loads.** `uint4` (128-bit) per thread, always. A
  32-bit load per thread leaves half the bandwidth on the table (Chapter 03).
- **The activation vector `x` is tiny** (`K` floats) and read by every block.
  It belongs in shared memory or L2, and it does not count against your
  bandwidth budget.
- **One block per output row wastes the machine** when `M` is small. Split `K`
  across blocks and combine, or assign multiple rows per block -- this is
  Chapter 07's reduction decomposition.
- **Scales are a second stream.** Align them, vectorize them, and make sure
  they are not issuing a separate dependent load per group.
- **The reduction at the end** is Chapter 07: shuffles, no shared memory round
  trip.

The gate for your kernel is 80% of the copy ceiling measured on *packed bytes
plus metadata bytes*. If you are below it, profile and find the extra traffic;
if you are at it, stop.

---

## 6. int8 tensor cores, for the other regime

When batch makes you compute-bound, integer tensor cores exist:

- `dp4a`: four-way int8 dot product with int32 accumulate, on CUDA cores.
  Available since Pascal, useful when tensor cores are busy or absent.
- `mma.sync` with `.s8` or `.u8` operands and `.s32` accumulate: roughly 2x
  fp16 tensor throughput.
- `.s4` on Turing and Ampere, 4x fp16 -- deprecated on Hopper, which is a
  useful lesson about building on narrow hardware features.
- fp8 (`e4m3`, `e5m2`) on Hopper, which is usually a better answer than int8
  for activations because it keeps dynamic range.

The complication is that the scales are per-channel, so you cannot apply them
inside the accumulation -- you accumulate in int32 and apply scales during
epilogue, where `y = (acc * s_w * s_x) + bias`. That epilogue is where the
fusion opportunities are (Chapter 10), and where most of the accuracy is won or
lost.

---

## 7. The KV cache is the other half

At long context, the KV cache rivals or exceeds the weights:

```
13B model, 40 layers, 40 heads, d = 128, fp16, context 8192, batch 1:
  2 (K and V) * 40 layers * 8192 * 40 * 128 * 2 bytes = 6.7 GB
```

And unlike weights, it grows with every token and with batch size. Decode reads
all of it every step, so it is bandwidth on exactly the same footing as the
weights.

Quantizing it to int8 or fp8 halves that traffic. It is more delicate than
weight quantization because the cache is written incrementally and the error
compounds across steps, and because per-token scales mean the quantization
granularity interacts with the attention tiling. Common practice: per-token,
per-head scales; keep the most recent window in fp16.

PagedAttention's non-contiguous block layout interacts with this too -- your
attention kernel now gathers KV blocks through an index table, which is
Chapter 22's problem.

---

## 8. Accuracy, enough to not be dangerous

You are a kernel engineer, not a quantization researcher, but you need enough
to tell "my kernel is wrong" from "quantization is lossy".

**Outliers dominate.** A few channels in transformer activations have
magnitudes 100x the rest, and a per-tensor scale spends its entire range on
them. This is the reason per-channel and per-group exist, and the reason
SmoothQuant migrates scale from activations into weights.

**Weights quantize much better than activations.** W4A16 (4-bit weights,
fp16 activations) is nearly lossless at 7B scale. W8A8 needs care. W4A4 is a
research problem.

**Test with the right instrument.** Perplexity on a held-out set, not a diff
against fp16 outputs. A kernel bug and a quantization error look identical in
an element-wise comparison, and completely different in perplexity. For
*kernel* correctness, compare against a dequantize-then-fp16-GEMM reference
using the same quantized values -- that isolates your kernel from the
quantization scheme entirely, and it is what the exercise harness does.

---

## 9. Common failure modes

**Correct for the first group, wrong after.** Group indexing: `k / group_size`
versus `k % group_size`, or a scale stride that assumes contiguity it does not
have.

**Correct for some rows, wrong for others.** Nibble order. Low nibble and high
nibble swapped shows up as every other element wrong.

**Small, uniform relative error everywhere.** Probably not a bug -- that is
quantization. Check against a dequantize-then-GEMM reference before debugging.

**Far below the copy ceiling.** In order of likelihood: scalar instead of
vectorized weight loads; scales re-read per element instead of per group;
uncoalesced access because the packing layout fights the thread mapping; too
few blocks to saturate.

**Unpack showing up in the profile.** On a memory-bound kernel the unpack
should be entirely hidden. If it is not, either you have too few warps to hide
it (occupancy) or your unpack is genuinely too long (layout).

**Great at `M = 4096`, terrible at `M = 128`.** Not enough parallelism. Split
`K`.

---

## 10. Exercises

### 21.1 - Find your crossover

For a 7B model on your GPU:

(a) Bandwidth-bound time per token at batch 1, fp16, int8, int4.
(b) Compute-bound time for a batch of `B` tokens, fp16 tensor cores.
(c) At which `B` do they cross? That is where decode becomes prefill.
(d) Redo (c) for int4 weights. Does the crossover move up or down, and why is
    the direction counter-intuitive at first?
(e) What does this say about continuous batching as a serving strategy?

### 21.2 - Group size is a bandwidth decision

For `g` = 32, 64, 128, 256, with fp16 scale and fp16 zero per group:

(a) Metadata bytes as a fraction of packed weight bytes.
(b) Token rate impact for each.
(c) At which `g` does metadata exceed 10% of traffic?
(d) If scales were fp8 or e8m0 instead, how does the table change?
(e) Given that accuracy improves monotonically with smaller `g`, where would
    you set it, and what would you need to measure to defend that?

### 21.3 - Design the packing format

(a) Write out a nibble layout for 8 int4 values in a `uint32` such that two
    fp16 values can be produced with one mask and one OR.
(b) Count instructions per 8 weights for your layout versus the naive
    sequential layout.
(c) At 80% of your copy ceiling, how many instructions per 8 weights can you
    afford before unpack stops being free? (This is the number that decides
    whether any of this matters on your hardware.)
(d) Work out the `0x6400` trick: show that `0x6400 | n` as fp16 equals
    `1024 + n` for `n` in `[0, 1024)`, and show where the `-1024` goes.
(e) Does the same trick work for bf16? For int8 to fp16?

### 21.4 - The roofline, before and after

(a) Arithmetic intensity of fp16 GEMV.
(b) Of int4 GEMV with `g = 128`.
(c) Plot both on your GPU's roofline.
(d) Both are far left of the ridge. So why is int4 4x faster -- what exactly
    moved?
(e) At what bit width would GEMV become compute-bound? Is that a sensible
    number? What does that tell you about the long-run limit of this approach?

### 21.5 - Quantize something by hand

Take a `[4096, 4096]` fp16 matrix of normally distributed values with 1% of
channels scaled by 50:

(a) Quantize to int4 per-tensor. Max and mean relative error.
(b) Per-channel. Same statistics.
(c) Per-group, `g = 128`. Same.
(d) Plot error against metadata bytes for all three.
(e) Now scale the outlier channels down by 50 and fold that into the next
    layer's weights (SmoothQuant). Re-measure. Explain the result in terms of
    where the quantization range is being spent.

### 21.6 - KV cache arithmetic

For a model you care about:

(a) KV cache size as a function of context length and batch.
(b) At which context does it exceed the weights?
(c) Bandwidth per decode step from weights, from cache, total.
(d) At context 32768, which dominates?
(e) If you could quantize only one of the two, which, and at what context
    length does the answer flip?

---

## Exercises to run

- [`21-int4-gemv`](../exercises/21-int4-gemv) - packed int4 weights, grouped
  scales. Bar: within 2e-2 of a dequantize-then-GEMM reference, >= 80% of your
  copy ceiling on packed bytes.

---

## What you should now believe

- Batch-1 decode is bandwidth-bound by roughly 200x. The tensor cores are idle
  and nothing in the inner loop matters except bytes.
- Quantization is a bandwidth optimization that happens to also save memory.
- The only honest metric for a decode kernel is percent of copy ceiling on
  packed bytes.
- Prefill and decode are different kernels with different bottlenecks; the
  crossover is a number you can compute.
- Group size trades accuracy against metadata traffic, and the traffic side is
  quantifiable.
- The packing layout exists to serve the unpack instruction sequence, not the
  human reading the file.
- The KV cache is the other bandwidth consumer and it eventually wins.

Next: [Chapter 22 - Sparsity, MoE and ragged batching](22-sparse-moe.md), where
the data stops being rectangular.
