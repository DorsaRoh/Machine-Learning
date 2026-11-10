# Chapter 15 - Numerics and precision

> Every kernel you have written has been quietly making numerical decisions.
> This chapter makes them explicit, because "my loss is NaN" and "my kernel
> disagrees with PyTorch in the 4th decimal" are kernel-engineering problems,
> and the person who can answer them is rare.

---

## 1. Floating point, briefly and precisely

```
value = (-1)^sign * 2^(exponent - bias) * 1.mantissa
```

| Format | Sign | Exp | Mantissa | Max | Min normal | eps (2^-m) |
|---|---|---|---|---|---|---|
| FP64 | 1 | 11 | 52 | 1.8e308 | 2.2e-308 | 2.2e-16 |
| FP32 | 1 | 8 | 23 | 3.4e38 | 1.2e-38 | 1.2e-7 |
| TF32 | 1 | 8 | 10 | 3.4e38 | 1.2e-38 | 9.8e-4 |
| BF16 | 1 | 8 | 7 | 3.4e38 | 1.2e-38 | 7.8e-3 |
| FP16 | 1 | 5 | 10 | 65504 | 6.1e-5 | 9.8e-4 |
| FP8 E4M3 | 1 | 4 | 3 | 448 | 1.5e-2 | 1.3e-1 |
| FP8 E5M2 | 1 | 5 | 2 | 57344 | 6.1e-5 | 2.5e-1 |

Two independent axes, and conflating them causes most precision bugs:

- **Exponent bits -> range.** Can you represent the number at all?
- **Mantissa bits -> precision.** How finely near that magnitude?

**BF16 and FP16 are both 16 bits and are not interchangeable.** BF16 has FP32's
exponent (huge range, coarse steps); FP16 trades 3 exponent bits for 3 mantissa
bits (finer steps, overflows at 65504). For deep learning, **range wins**,
which is why BF16 took over: attention scores and gradients routinely exceed
65504, and FP16 requires loss scaling and constant care to avoid it.

### The two rules that matter

1. **Relative spacing is constant; absolute spacing is not.** Near 1.0, fp32
   steps are ~1e-7. Near 1e6, they are ~0.06. `1e6 + 0.01 == 1e6` in fp32.
2. **Addition is not associative.** `(a + b) + c != a + (b + c)`. This single
   fact is why GPU results are not bit-reproducible, why atomics are
   non-deterministic, and why "my kernel disagrees with PyTorch" is usually not
   a bug.

---

## 2. Catastrophic cancellation

Subtracting nearly-equal numbers destroys significant digits. The errors were
already there, in the low bits; subtraction promotes them.

```
a = 1.0000001e8   (fp32: ~7 good digits)
b = 1.0000000e8
a - b = 10.0       but only ~0 significant digits survive
```

Where it shows up in ML kernels:

- `var = E[x^2] - E[x]^2` (Chapter 10). The classic.
- `softmax` without max subtraction: `exp` of a large number overflows first,
  so you never even reach the cancellation.
- Numerical gradients: `(f(x+h) - f(x))/h` is cancellation by construction.
- Any "compute two big things and subtract" formula.

**The fix is always algebraic, not numerical.** Rearrange so you never subtract
near-equal quantities: Welford instead of the sum-of-squares identity,
`log1p(x)` instead of `log(1+x)`, `expm1(x)` instead of `exp(x)-1`.

---

## 3. Accumulation

Summing `n` values in fp32:

```
sequential   : error grows as O(n * eps)      -- bad for large n
pairwise/tree: error grows as O(log n * eps)  -- what a GPU does naturally
Kahan        : error ~ O(eps), independent of n
```

**The GPU's tree reduction is more accurate than a CPU's naive loop** (Chapter
07 section 6). When your GPU disagrees with a reference implementation, check
which one is actually right before assuming it is you.

**Kahan summation** keeps a running compensation term:

```cuda
float sum = 0, c = 0;
for (...) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;   // the bits that were lost
    sum = t;
}
```

Four operations instead of one. On a memory-bound reduction that is **free** --
you are waiting for memory anyway. Worth knowing as a tool for when accuracy
matters more than the last 5% of bandwidth.

**The practical rule for ML kernels: inputs may be narrow, accumulators are
fp32.** Tensor cores enforce this; your hand-written kernels should too. An
fp16 accumulator over K = 4096 loses the sum entirely, because once the running
total is ~100 and the addends are ~0.01, fp16's step size at 100 is 0.0625 and
every addend rounds to nothing.

---

## 4. Mixed precision in practice

The standard training recipe:

```
weights       : fp32 master copy
forward/back  : bf16 (or fp16 with loss scaling)
accumulation  : fp32 (inside GEMM, and for optimizer state)
optimizer     : fp32
```

Why a master copy: a weight update of `1e-7` applied to a bf16 weight of `1.0`
does nothing at all -- bf16's step near 1.0 is ~0.0078. The update vanishes and
training stalls. The fp32 master copy accumulates small updates correctly and
is cast to bf16 for the forward pass.

**Loss scaling** (FP16 only): gradients are small and underflow fp16's minimum
normal of 6e-5. Multiply the loss by ~65536 before backward, divide the
gradients by the same before the update. Dynamic loss scaling adjusts the
factor when it detects infs. With BF16 none of this is needed, which is most of
why BF16 won.

**FP8** (Hopper): E4M3 for forward activations and weights, E5M2 for gradients
(which need the extra range). Requires per-tensor scaling factors, tracked and
updated as the distribution shifts. This is bookkeeping your kernel has to
carry, and it is why FP8 GEMM APIs take scale arguments.

---

## 5. Determinism

A kernel is deterministic if the same input gives bit-identical output every
run. Sources of non-determinism, in order of frequency:

1. **Atomics.** Order of combination varies with scheduling.
2. **Grid-size-dependent reduction trees.** Different block counts give
   different trees.
3. **Autotuning / library heuristics** picking different kernels run to run.
4. **cuDNN algorithm selection** that benchmarks at runtime.

Determinism costs performance -- typically 10-30% -- because it rules out the
fastest atomic-based strategies. The choice is real:

- **You need it for:** debugging, regression tests with tight tolerances,
  scientific reproducibility, and anything subject to audit.
- **You can skip it for:** training throughput, where the noise is far below
  what SGD already injects.

In PyTorch: `torch.use_deterministic_algorithms(True)` plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. For your own kernels: fixed reduction trees,
fixed grid sizes, no float atomics.

### Fixed-point accumulation

If you need both determinism and atomics, accumulate in a scaled integer:

```cuda
atomicAdd(&fixed_sum, (long long)(value * (1 << 20)));
```

Integer addition is associative, so this is order-independent and exactly
reproducible. You trade dynamic range for determinism. This is a real technique
in gradient accumulation for reproducible training.

---

## 6. Fast math

```bash
nvcc -use_fast_math        # the whole bundle
```

which implies:

| Flag | Effect | Risk |
|---|---|---|
| `-ffast-math` style reassociation | reorders FP ops | changes results |
| `--prec-div=false` | approximate division | ~2 ulp |
| `--prec-sqrt=false` | approximate sqrt | ~2 ulp |
| `--ftz=true` | denormals flush to zero | underflow becomes 0 |
| intrinsic substitution | `expf` -> `__expf` etc. | ~2-4 ulp |

The intrinsics individually:

```cuda
__expf(x)    // ~2 ulp, hardware SFU, ~4 cycles
expf(x)      // <1 ulp, software, ~20+ cycles
__fdividef(a,b), __sinf, __cosf, __logf, __powf
__frcp_rn, __fsqrt_rn, rsqrtf   // rsqrtf is a single hardware instruction
```

**Use the intrinsics explicitly rather than enabling `-use_fast_math`
globally.** A global flag changes the numerics of code you did not think
about -- including your correctness checks. Explicit `__expf` in the one place
it matters is a decision; a compiler flag is a wish.

For softmax and attention, `__expf` is universally used and the accuracy is
fine, because the result is immediately normalised.

---

## 7. Debugging numerical problems

A procedure, in order:

1. **Compare against float64 on the CPU.** Not against another fp32
   implementation -- you need ground truth.
2. **Shrink the input.** n = 16. Print everything. Most "numerical" bugs are
   indexing bugs.
3. **Find the first bad element.** Is it element 0, or element 1024? Position
   is information: element 0 suggests a formula error, element 1024 suggests a
   tile boundary.
4. **Check the error magnitude against theory.** `sqrt(K) * eps` for a dot
   product of length K. If your error is 1e-5 and theory says 1e-5, there is no
   bug.
5. **Check for NaN/Inf and find where they first appear.** `isnan` guards plus
   `printf`, or `compute-sanitizer --tool initcheck` for uninitialised reads
   (which often surface as NaN).
6. **Widen the accumulator.** If fp64 accumulation fixes it, you have a
   precision problem. If it does not, you have a logic problem. This single
   test separates the two categories and is worth doing early.

### Error budget you should expect

| Operation | Expected relative error (fp32) |
|---|---|
| Elementwise | ~1e-7 |
| Dot product, length K | ~sqrt(K) * 1e-7 |
| GEMM, K = 4096 | ~6e-6 |
| Tree reduction, n = 1e8 | ~3e-6 |
| Sequential sum, n = 1e8 | ~1e-2 or worse |
| fp16 GEMM, fp32 accumulate, K = 4096 | ~1e-3 |
| fp16 GEMM, fp16 accumulate | garbage |

If your measured error matches the table, stop looking for a bug.

---

## 8. Common failure modes

**NaN appears after N steps of training.** Overflow somewhere, usually fp16
attention scores or an unscaled gradient. Find the first NaN with a hook.

**Loss is fine but the model does not learn.** Updates underflowing into a
narrow weight format. Check for an fp32 master copy.

**Results differ run to run.** Atomics. Expected.

**Your kernel differs from PyTorch by 1e-6.** Not a bug. Different accumulation
order.

**Your kernel differs from PyTorch by 1e-2.** A bug, or fp16 accumulation, or a
missing max-subtraction.

**Variance is negative.** Cancellation. Welford.

**Correct in fp64, wrong in fp32.** A precision problem; find the cancellation
or the long accumulation.

**Wrong only for large inputs.** Range, not precision. Check for overflow and
for index overflow -- `int` arithmetic on a 3e9-element array silently wraps,
and that is the other half of "wrong only for large inputs".

---

## 9. Exercises

### 15.1 - Representability

(a) Largest integer exactly representable in fp32? In fp16? In bf16?
(b) `16777216.0f + 1.0f` in fp32: what do you get, and why?
(c) Spacing between consecutive fp32 values near 1.0? Near 1e6? Near 1e-30?
(d) Smallest fp16 normal; smallest fp16 subnormal. What does `--ftz=true` do to
    the second one?
(e) `float x = 0.1f; (x + x + x) == 0.3f`? Why not? Which is closer to the real
    0.3: `0.1f*3` or `0.1f+0.1f+0.1f`?

### 15.2 - Cancellation, measured

(a) For `x ~ N(1000, 1)`, D = 4096: compute the variance three ways -- the
    sum-of-squares identity in fp32, Welford in fp32, and float64 ground truth.
    Report relative errors.
(b) Find the mean offset at which the fp32 identity loses half its digits.
(c) Does fp64 accumulation in the identity fix it? At what cost on your GPU
    (check your chip's fp64 rate)?
(d) Does fp32 Welford beat fp64-naive? That comparison -- better algorithm in
    narrow precision versus worse algorithm in wide precision -- is the general
    lesson.

### 15.3 - Accumulation order

Sum 1e8 fp32 values of magnitude ~1.0:

(a) Sequentially on the CPU. Relative error vs float64?
(b) Pairwise/tree. Error?
(c) Your GPU reduction from `07-reduce`. Error?
(d) Kahan on the CPU. Error?
(e) Now sum values of *wildly* varying magnitude (mix 1e8 and 1e-8). Redo all
    four. Which degrades most, and why?
(f) Sort by magnitude first, then sum ascending. Better or worse? Explain.

### 15.4 - The fp16 accumulator experiment

Using your `13-tensor-core-gemm`:

(a) fp32 accumulate, K = 4096. Relative error vs float64.
(b) fp16 accumulate, K = 4096. Error.
(c) fp16 accumulate, K = 128. Error. Why is it so much better?
(d) At what K does fp16 accumulation lose its first significant digit? Derive
    it, then verify.
(e) Does the input *distribution* change your answer? Try all-positive values
    versus zero-mean.

### 15.5 - Build a precision lab

In `15-precision-lab`, implement one computation in fp32, tf32, bf16, fp16 and
(if available) fp8, with configurable accumulator width, and measure error and
speed for each.

(a) Which combinations are Pareto-optimal on your GPU?
(b) Which would you pick for training? For inference? For a regression test?
(c) Construct an input that breaks fp16 but not bf16, and one that breaks bf16
    but not fp16. The second one is harder and is the whole argument for when
    fp16 is still worth using.

### 15.6 - Determinism audit

(a) Run your `07-reduce` atomic version 1000 times. How many distinct results?
    What is the spread, in ulps?
(b) Run the two-kernel version 1000 times. Distinct results?
(c) Change the grid size and re-run. Same answer?
(d) Implement fixed-point accumulation. Verify bit-exactness. Measure the cost.
(e) Write the one-paragraph policy you would put in a team's kernel guidelines:
    when is non-determinism acceptable, and what must be true before it ships?

---

## Exercises to run

- [`15-precision-lab`](../exercises/15-precision-lab) - no performance gate;
  the deliverable is a written explanation of each failure.

---

## What you should now believe

- Range and precision are different axes, and BF16 beat FP16 on range.
- Addition is not associative, so GPU results are not reproducible by default.
- Accumulate wide, always. fp16 accumulation over long K is useless.
- Cancellation is fixed by algebra, not by wider types.
- Tree reduction is more accurate than sequential summation.
- Determinism is a deliberate trade, not a default.
- Use intrinsics explicitly; do not flip `-use_fast_math` globally.

Next: [Chapter 16 - Profiling](16-profiling.md).
