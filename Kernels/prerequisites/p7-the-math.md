# P7 - The math you need

> Less than you fear. This course is not a machine learning course and it is
> not a numerical analysis course. You need to know what five operations *are*,
> well enough to recognise one when it appears in a loop, because every kernel
> in the main course is one of them or a fusion of several.

---

## 1. Vectors and element-wise operations

A **vector** is a list of numbers. In memory it is an array; the word "vector"
just means we intend to do arithmetic on it as a whole.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
```

**Element-wise** means apply the operation at each position independently:

```
a + b     = [6, 8, 10, 12]
a * b     = [5, 12, 21, 32]        (element-wise, NOT matrix multiply)
2 * a     = [2, 4, 6, 8]           (scalar broadcast)
relu(a)   = [max(0,1), max(0,2), ...] 
```

In code:

```cpp
for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
```

**Every element is independent of every other.** No element needs to know
anything about its neighbours. That property is what makes this the first thing
anyone writes on a GPU: `02-vector-add` is this loop with the `for` replaced by
"one thread per `i`".

It is also the cheapest possible operation per byte moved: 3 arrays touched,
1 arithmetic operation. Hold that thought until P8.

**SAXPY** — "single-precision a times x plus y" — is the named version:

```
y = alpha * x + y
```

It is `02-saxpy`, and it is the canonical memory-bound kernel.

---

## 2. Dot product: the first reduction

A **dot product** multiplies two vectors element-wise and sums the results,
giving one number:

```
a · b = 1*5 + 2*6 + 3*7 + 4*8 = 70
```

```cpp
float dot = 0.0f;
for (int i = 0; i < n; i++) dot += a[i] * b[i];
```

Notice what changed from section 1: there is now a single `dot` that every
iteration reads and writes. The iterations are **no longer independent**.

That dependency is the entire difficulty of Chapter 07. On a GPU you cannot
have 10,000 threads all adding into one variable — they would collide, and they
would serialise. The solution is to exploit the fact that addition is
associative (mostly — see P4 section 5) and sum in a *tree*:

```
level 0:  a0  a1  a2  a3  a4  a5  a6  a7
level 1:    s0      s1      s2      s3        (pairs)
level 2:        t0              t1
level 3:                u0                    (the answer)
```

`n` elements in `log2(n)` levels instead of `n` steps. Eight elements in three
levels. A million in twenty.

**This is a "reduction": many values in, one value out.** Sum, maximum,
minimum, product, count — all reductions, all the same tree.

---

## 3. Matrix multiply

A matrix is a 2D grid of numbers (P3 for how it is stored). Matrix multiply:

```
C = A B      where A is [M, K], B is [K, N], C is [M, N]
```

**Each element of C is a dot product**: row `i` of A with column `j` of B.

```
C[i][j] = sum over k of  A[i][k] * B[k][j]
```

In flat row-major indexing:

```cpp
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[i * K + k] * B[k * N + j];
        C[i * N + j] = acc;
    }
```

Three loops. Read the index expressions carefully — `A` uses stride `K`, `B`
uses stride `N`, `C` uses stride `N`. Getting these right is P3, and it is
genuinely most of what makes `11-gemm-ladder` work.

**The dimensions must agree:** A's columns must equal B's rows. That shared
dimension `K` is the one summed over and it disappears from the result.

Two facts to carry forward:

**The work is `2*M*N*K` floating-point operations** — one multiply and one add
per `k`, for each of the `M*N` outputs. You will compute this constantly.

**Every element of A is used `N` times and every element of B is used `M`
times.** That reuse is why matrix multiply can be compute-bound while vector
add cannot, and it is the thing tiling exists to exploit. P8 makes this precise.

**Transpose** swaps rows and columns: `A^T[j][i] = A[i][j]`. It appears
constantly because `B` is often stored such that you want its rows rather than
its columns — and because, as P3 section 5 showed, it is a pure indexing
exercise with a performance trap in it.

---

## 4. Softmax

Softmax turns a list of arbitrary numbers into a probability distribution —
all positive, summing to 1:

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

```cpp
float sum = 0.0f;
for (int i = 0; i < n; i++) sum += std::exp(x[i]);
for (int i = 0; i < n; i++) out[i] = std::exp(x[i]) / sum;
```

**That code is wrong**, and it is wrong for a reason you already know. If any
`x[i]` is large — say 1000 — then `exp(1000)` overflows to infinity (P4 section
6), the sum becomes infinity, and every output is `inf/inf` = NaN.

The fix is to subtract the maximum first:

```
softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

This is mathematically identical — the factor `exp(-max)` cancels top and
bottom — but now the largest exponent is `exp(0) = 1` and nothing overflows.

So softmax is really **three passes**: find the max, sum the exponentials,
divide. Each pass reads the whole row.

And that is the setup for one of the better ideas in the course. Chapter 10
shows how to fuse the first two passes by carrying a running `(max, sum)` pair
and correcting it when a new maximum appears. Chapter 19 then uses that same
trick to make attention work without ever materialising its `N x N`
intermediate — which is FlashAttention. **The entire algorithm descends from
needing to avoid the overflow in the naive code above.**

---

## 5. Normalisation

**Mean and variance**, for a list:

```
mean = (1/n) * sum of x_i
var  = (1/n) * sum of (x_i - mean)^2
std  = sqrt(var)
```

**LayerNorm** standardises a row, then applies a learned scale and shift:

```
y_i = gamma_i * (x_i - mean) / sqrt(var + eps) + beta_i
```

`eps` is a small constant (say 1e-5) preventing division by zero when the row
is constant.

Two things worth noticing now, because both become exercises:

**The algebraic shortcut is a trap.** `var = E[x^2] - E[x]^2` lets you compute
mean and variance in one pass instead of two, which is tempting on a GPU where
passes cost bandwidth. It is also catastrophically unstable when the mean is
large (P4 section 5), and `10-layernorm` feeds you data with mean 1000
specifically to catch it.

**RMSNorm** drops the mean entirely:

```
y_i = gamma_i * x_i / sqrt( mean(x^2) + eps )
```

One reduction instead of two, and modern transformers mostly use it.

---

## 6. Attention, in advance

You do not need this until Chapter 19, but seeing it once now means it is not
new then. Attention is:

```
S = Q K^T / sqrt(d)        a matrix multiply
P = softmax(S)             a softmax, row by row
O = P V                    another matrix multiply
```

Two matrix multiplies with a softmax in between. **Every piece is something in
this chapter.** When Chapter 19 says it derives FlashAttention from the
roofline, it is combining section 3, section 4 and P8 — nothing else.

---

## 7. Derivatives, the minimum

Chapter 20 is about backward passes. You need one idea.

A derivative is a rate of change: if `y = f(x)`, then `dy/dx` says how much `y`
moves when `x` moves a little. Training a network means computing, for every
parameter, how much the loss would change if that parameter moved — and
stepping in the direction that lowers it.

The **chain rule** composes them. If `y = f(x)` and `z = g(y)`:

```
dz/dx = (dz/dy) * (dy/dx)
```

Backpropagation is the chain rule applied through a network, right to left. You
are handed `dL/dy` (how much the loss cares about your output) and must produce
`dL/dx` (how much it cares about your input).

The handful you will actually need:

```
y = x^2          ->  dy/dx = 2x
y = a*x          ->  dy/dx = a
y = exp(x)       ->  dy/dx = exp(x)
y = sum of x_i   ->  dy/dx_i = 1        (a sum's gradient is a broadcast)
y_i = x_i        ->  element-wise stays element-wise
C = A B          ->  dA = dC B^T,  dB = A^T dC
```

That last line is the one that matters, and the pattern it reveals is Chapter
20's opening table: **a broadcast forwards becomes a reduction backwards, and a
reduction forwards becomes a broadcast backwards.** If you know that, you can
usually work out the shape of a backward pass before doing any calculus.

---

## 8. Exercises

Paper for the derivations, code for the rest. Keep your implementations — they
are the CPU references you will check your GPU kernels against.

### P7.1 - By hand

```
a = [1, 2, 3],  b = [4, 5, 6]

A = [[1, 2],       B = [[5, 6],
     [3, 4]]            [7, 8]]
```

(a) `a + b`, `a * b` element-wise, `a · b`.
(b) `A B` by hand. Then `B A`. Are they equal? What does that tell you?
(c) `A^T`. `(A B)^T`. Is it `A^T B^T` or `B^T A^T`? Verify numerically.
(d) `softmax([1, 2, 3])`. Then `softmax([101, 102, 103])`. Equal? Why?
(e) `softmax([1, 2, 3])` computed naively in fp32 after adding 100 to every
    element — what goes wrong, and at roughly what magnitude does it start?

### P7.2 - Write the references

In plain C++, with `double` accumulators:

(a) `double dot(const float* a, const float* b, int n)`.
(b) `void matmul(const float* A, const float* B, float* C, int M, int N, int K)`.
(c) `void softmax(const float* x, float* out, int n)` — numerically stable.
(d) `void layernorm(const float* x, const float* g, const float* b, float* out,
    int n, float eps)` — two-pass.
(e) Test each against values you computed by hand in P7.1.
(f) Why `double` accumulators when the data is `float`? (P4 section 5.)

### P7.3 - Count the work

For `M = N = K = 1024`:

(a) FLOPs in the matmul.
(b) Bytes read and written, assuming each element is touched once.
(c) FLOPs per byte.
(d) Same three for a vector add of 1024*1024 elements.
(e) The ratio between (c) and its vector-add equivalent. Which operation has
    more arithmetic available per byte moved?
(f) Predict which one a GPU can run closer to its peak arithmetic rate. You
    have just done the core calculation of Chapter 01.

### P7.4 - Reuse

In the matmul of P7.3:

(a) How many times is each element of `A` read, in the naive triple loop?
(b) How many times is each element of `B` read?
(c) If every one of those reads came from RAM, how many bytes would move?
(d) Compare with (b) of P7.3. What is the ratio?
(e) What would have to be true for the naive loop to move only the (b) amount?
(f) That is what tiling achieves. Without knowing how it works, state what it
    must be doing.

### P7.5 - The gradient shapes

Without doing any calculus, just using the broadcast/reduction rule from
section 7:

(a) Forward: `y = sum(x)` over a length-`n` vector. What shape is `dL/dx`, and
    what is in it?
(b) Forward: `y_i = x * w_i` (one `x`, many `w`). What does `dL/dx` require?
(c) Forward: `C = A B`. What shapes are `dA` and `dB`?
(d) In which of (a)-(c) does a *reduction* appear in the backward pass?
(e) Chapter 20 claims reductions in the backward pass are where atomics and
    non-determinism come from. Given (d), explain why.

---

## What you should now believe

- Element-wise operations have fully independent iterations, which is why they
  are the first thing you parallelise and the least interesting thing you run.
- A dot product is a reduction: many in, one out, done as a tree in
  `log(n)` levels because the iterations are not independent.
- Matrix multiply is `M*N` dot products, costs `2*M*N*K` FLOPs, and reuses
  every input many times — which is why it can be compute-bound.
- Naive softmax overflows; subtracting the max is mathematically free and
  numerically essential. FlashAttention grows out of fixing that in one pass.
- LayerNorm is two reductions plus an element-wise pass, and the one-pass
  algebraic shortcut is a numerical trap.
- Backward passes transpose the dataflow: broadcasts become reductions and
  reductions become broadcasts.

Next: [P8 - Performance arithmetic](p8-performance-arithmetic.md), the last
one, and the direct run-up to Chapter 01.
