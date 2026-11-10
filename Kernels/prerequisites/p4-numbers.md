# P4 - How numbers actually work

> `0.1 + 0.2 != 0.3` is usually presented as a funny quirk. It is not a quirk,
> it is the defining constraint of numerical computing, and on a GPU — where a
> sum is computed in a different order every time — it decides whether your
> results are reproducible at all.

---

## 1. Binary, briefly

Decimal uses powers of ten. Binary uses powers of two.

```
decimal 13  =  1*8 + 1*4 + 0*2 + 1*1  =  1101 in binary
```

Useful powers to know on sight, because they appear as array sizes constantly:

```
2^8  = 256        2^16 = 65,536         2^20 = 1,048,576  (~1 million)
2^10 = 1024       2^24 = 16,777,216     2^30 = 1,073,741,824 (~1 billion)
```

Two operations you will see in kernel code:

```cpp
x << 1    // shift left: multiply by 2
x >> 1    // shift right: divide by 2 (integer)
x & 31    // bitwise and: keeps the low 5 bits. Same as x % 32.
```

`x & 31` being `x % 32` is a trick you will meet in Chapter 05: for powers of
two, masking is the same as a remainder and much cheaper. `threadIdx.x & 31`
gives a thread's lane within its warp.

---

## 2. Integers, and the edge they fall off

A 32-bit `int` holds roughly -2.1 billion to +2.1 billion. Exceed that and it
**wraps around**:

```cpp
int big = 2147483647;      // the maximum
big = big + 1;             // now -2147483648. Not an error. No warning.
```

This matters more than it sounds, because array indices are often `int` and
large arrays overflow them:

```cpp
int n = 100000;
int total = n * n;          // 10,000,000,000 -> overflows, becomes garbage
long long ok = (long long)n * n;   // correct
```

**The cast must happen before the multiply.** `(long long)(n * n)` is too late;
the overflow already happened in `int`.

In this course you will see `size_t` everywhere for exactly this reason:

```cpp
size_t idx = size_t(row) * width + col;
```

`size_t` is an unsigned 64-bit integer meant for sizes and indices. Use it when
the product could be large. Every harness in the main course does.

One gotcha with unsigned types: they never go below zero, they wrap.

```cpp
size_t n = 0;
for (size_t i = 0; i < n - 1; i++)   // n-1 is 18,446,744,073,709,551,615
```

An infinite loop, from a subtraction that looked harmless.

---

## 3. Floating point: the actual format

A `float` is 32 bits split into three fields:

```
 [ sign | exponent | mantissa ]
    1       8 bits    23 bits

value = (-1)^sign  x  1.mantissa  x  2^(exponent - 127)
```

It is scientific notation in binary. The mantissa gives you the digits, the
exponent gives you the scale.

The consequence of the split is the whole story:

**You get about 7 decimal digits of precision, at any magnitude.** Not 7
decimal places — 7 significant digits. `0.0001234567` and `1234567.0` are both
representable to about the same *relative* accuracy. The absolute gap between
representable numbers grows with the magnitude:

| Near | Gap between neighbouring floats |
|---|---|
| 1.0 | 0.00000012 |
| 1000.0 | 0.000061 |
| 1,000,000.0 | 0.0625 |
| 16,777,216.0 | 2.0 |

That last row deserves attention. **Above 2^24 = 16,777,216, a `float` cannot
represent consecutive integers.** So:

```cpp
float x = 16777216.0f;
x = x + 1.0f;              // still 16777216.0f
```

Adding 1 does nothing, forever. If you sum ten million values of 1.0f in a
float, you get 16,777,216 and the loop stops making progress. Chapter 15 opens
with a version of this.

The formats you will meet:

| Format | Bits | Mantissa | Exponent | Decimal digits | Max |
|---|---|---|---|---|---|
| `double` | 64 | 52 | 11 | ~16 | 1.8e308 |
| `float` | 32 | 23 | 8 | ~7 | 3.4e38 |
| `half` (fp16) | 16 | 10 | 5 | ~3 | 65504 |
| `bfloat16` | 16 | 7 | 8 | ~2 | 3.4e38 |

Look at the last two rows together. They are the same size, and they spend
their bits differently: `half` buys precision, `bfloat16` buys range.
`bfloat16` has `float`'s exponent field exactly, so anything that fits in a
float fits in a bfloat16 (badly). That single design choice is why bf16 largely
replaced fp16 for training — Chapter 20 section 6 tells that story, and you now
have everything you need to understand it.

---

## 4. Why `0.1 + 0.2 != 0.3`

In binary, 0.1 is a repeating fraction, exactly as 1/3 is in decimal. You
cannot write it down in finite digits. So the stored value is the nearest
representable float, which is slightly off, and the errors accumulate:

```cpp
float a = 0.1f + 0.2f;     // 0.30000001192092896
float b = 0.3f;            // 0.30000001192092896? Not necessarily.
a == b;                    // may be false
```

**Therefore: never compare floats with `==`.** Compare with a tolerance:

```cpp
if (std::fabs(a - b) < 1e-6f) { /* close enough */ }
```

Better, for values whose magnitude you do not know in advance, compare
*relative* error:

```cpp
if (std::fabs(a - b) / std::fabs(b) < 1e-6f) { ... }
```

This is precisely what `check::eq_double` and the main course's
`bench::compare` do, and why every exercise in this course has a stated
tolerance rather than demanding exactness. When `19-flash-fwd` says "within
2e-3", that number is a judgement about what fp16 arithmetic can deliver, not
a slack allowance.

---

## 5. Addition is not associative, and this is the big one

In real arithmetic, `(a + b) + c == a + (b + c)`. In floating point it is
**not**, because each operation rounds:

```cpp
float a = 1e10f, b = -1e10f, c = 1.0f;

(a + b) + c   // (0) + 1       = 1.0
a + (b + c)   // 1e10 + (-1e10 + 1)
              // -1e10 + 1 rounds back to -1e10 (the 1 is lost)
              // so the result is 0.0
```

Same three numbers, two groupings, completely different answers. One of them
has lost 100% of the information.

Two named phenomena come out of this:

**Catastrophic cancellation.** Subtracting two nearly equal large numbers
annihilates the significant digits. The classic trap is computing variance as
`E[x^2] - E[x]^2`: with data near 1000, both terms are around 1,000,000 and
their difference is around 1, so you have subtracted away six of your seven
digits. Chapter 15 and the `10-layernorm` exercise test exactly this, which is
why that exercise feeds you data with mean 1000.

**Order-dependent summation.** Summing a million floats left-to-right
accumulates error, because the running total grows large while the addends stay
small, and eventually each addend is below the precision of the total. Summing
in a tree — pairs, then pairs of pairs — keeps the magnitudes comparable and is
far more accurate.

Now here is why this is a *GPU* topic rather than a numerical-analysis topic:

**A parallel reduction is a tree.** Chapter 07's reduction does not sum
left-to-right; it sums in whatever order the threads finish. So:

- Your GPU result will differ from your CPU result. That is expected, and often
  the GPU is *more* accurate.
- If the order varies between runs — which it does whenever `atomicAdd` on
  floats is involved — your program is **not reproducible**. Chapter 15 and
  Chapter 20 both make you decide deliberately whether you can accept that.

So the rule you will see everywhere in this course:

> **Accumulate in a wider type than you store.** Inputs in fp16, accumulator in
> fp32. Inputs in fp32, reference in fp64.

Every reference implementation in the main course computes in `double` for
exactly this reason.

---

## 6. Special values

```cpp
float inf  = 1.0f / 0.0f;        // infinity. Not a crash.
float ninf = -1.0f / 0.0f;
float nan  = 0.0f / 0.0f;        // Not a Number
```

**NaN is contagious**: any arithmetic involving NaN produces NaN. So one bad
value early in a computation turns the entire output into NaN, which is
actually helpful — it is loud.

**NaN is not equal to anything, including itself:**

```cpp
nan == nan      // false
std::isnan(nan) // true -- this is how you test
```

`x != x` being true is the classic idiom for "x is NaN", and you will see it in
harness code.

Where NaN comes from in this course, in order of frequency:

- `exp` of a large number → `inf`, then `inf - inf` or `inf / inf` → NaN. This
  is why softmax subtracts the row maximum first, and why `10-softmax` feeds
  you inputs in `[-10, 10]` to make the failure obvious.
- Dividing by a sum that came out zero — a fully masked attention row
  (Chapter 19 section 7).
- `sqrt` of a negative, from a variance that went slightly negative through
  cancellation (section 5).

When a kernel produces NaN, these three are where to look first.

---

## 7. Exercises

### P4.1 - Find the edge

Write a program that:

(a) Starts at `float x = 1.0f` and doubles it until `x + 1.0f == x`. Print the
    value. What power of two is it? Why that one?
(b) Starts at `float x = 1.0f` and halves it until `1.0f + x == 1.0f`. Print.
    This is machine epsilon.
(c) Repeats both for `double`. How do the answers relate to the mantissa bit
    counts in section 3?
(d) Prints `0.1f + 0.2f - 0.3f` with `%.20f`. What exactly is the error?

### P4.2 - Summation order

Make an array of 10 million `float`s, all `1.0f`.

(a) Predict the sum. Then compute it left-to-right in a `float` accumulator.
    What did you get, and why that specific number?
(b) Same data, accumulate in a `double`. Now?
(c) Same data, sum in pairs recursively (a tree). In a `float` accumulator?
(d) Now make the array `0.1f` instead. Repeat all three. Which is closest to
    the true answer, and how do you know what the true answer is?
(e) Which of these three is a GPU reduction most like?

### P4.3 - Cancellation

For data with mean `m` and a small spread:

(a) Compute variance as `E[x^2] - E[x]^2` in `float`.
(b) Compute it as `E[(x - mean)^2]` in `float`, two passes.
(c) Do both for `m` = 0, 10, 1000, 100000.
(d) Plot or tabulate the error of each against `m`. At which `m` does (a)
    become useless?
(e) Can (a) produce a *negative* variance? Construct an input where it does.
(f) What would you do about it if you were only allowed one pass over the data?
    (You do not have to solve it — just state the difficulty. Chapter 15 names
    the solution.)

### P4.4 - NaN hunting

(a) Compute `std::exp(100.0f)`. What do you get?
(b) Compute `std::exp(100.0f) / std::exp(100.0f)`. Mathematically this is 1.
    What do you actually get, and at which step was it lost?
(c) Now compute `std::exp(100.0f - 100.0f) / std::exp(100.0f - 100.0f)`. Why is
    this one fine?
(d) You have just derived the max-subtraction trick in softmax. Write the
    general rule in one sentence.
(e) Write a function that finds the index of the first NaN in an array. Which
    comparison do you use, and why not `== NAN`?

### P4.5 - The formats

(a) What is the largest integer a `float` can represent exactly? A `half`?
(b) A model has 7 billion parameters. Bytes in fp32, fp16, int8, int4?
(c) `half` maxes out at 65504. If you sum 100,000 values of 1.0 in fp16, what
    happens, and at what point?
(d) Why does `bfloat16` not have that problem for the *range*, and what does it
    give up instead?
(e) Given (c) and (d), state the rule about accumulator width in your own
    words.

---

## What you should now believe

- A `float` is 4 bytes giving ~7 significant digits, and above 2^24 it cannot
  count by ones.
- Never compare floats with `==`; compare against a tolerance, usually relative.
- Floating-point addition is not associative, so the order of a sum changes its
  value — and a parallel reduction is a different order by construction.
- Catastrophic cancellation destroys precision when you subtract nearly equal
  large numbers. `E[x^2] - E[x]^2` is the canonical trap.
- Accumulate in a wider type than you store. Always.
- NaN is contagious, is not equal to itself, and usually comes from `exp` of
  something large or a division by a zero sum.
- fp16 and bf16 are the same size and spend their bits differently: precision
  versus range.

Next: [P5 - Just enough C++](p5-cpp-for-cuda.md).
