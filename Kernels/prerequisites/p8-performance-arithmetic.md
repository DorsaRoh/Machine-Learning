# P8 - Performance arithmetic

> The central skill of this entire course is looking at a loop and saying "that
> should take 1.4 milliseconds" before running it. This chapter teaches that on
> a CPU, where you can check your answers today. Chapter 01 does the same thing
> with a GPU's numbers and calls it the roofline.

---

## 1. Units, and getting them right

Prefixes. Powers of ten for rates, powers of two for memory sizes — which is a
genuine ambiguity you will meet constantly:

| Prefix | Decimal | Binary |
|---|---|---|
| kilo | 10^3 = 1,000 | 2^10 = 1,024 |
| mega | 10^6 | 2^20 = 1,048,576 |
| giga | 10^9 | 2^30 |
| tera | 10^12 | 2^40 |

Convention, which this course follows: **bandwidth and FLOP rates use decimal;
memory capacities use binary.** So "900 GB/s" means 900 x 10^9 bytes per
second, while "16 GB of memory" means 16 x 2^30 bytes. Nobody is going to
announce which they mean; the 7% discrepancy is not worth worrying about for
estimates, but know it exists.

**Bytes versus bits.** 8 bits to a byte. Network and PCIe specs are usually in
bits per second, memory bandwidth in bytes per second. A capital B is bytes, a
lowercase b is bits. Getting this wrong is an 8x error, which is larger than
most optimisations.

**Time.**

```
1 second = 1,000 milliseconds (ms) = 1,000,000 microseconds (us) = 1e9 nanoseconds (ns)
```

Kernels are timed in milliseconds and microseconds. A launch overhead of "5 us"
is 0.005 ms, which matters when your kernel takes 0.02 ms — Chapter 17's whole
subject.

---

## 2. The two numbers that describe any loop

For any piece of code, ask two questions.

**How many bytes does it move?** Count every array element read from or written
to main memory, times the size of the element.

**How many floating-point operations does it do?** Count multiplies, adds,
etc. (A fused multiply-add is conventionally counted as 2 FLOPs, since it does
both.)

Worked, for vector add over `n` floats:

```cpp
for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
```

```
bytes  = read a (4n) + read b (4n) + write c (4n) = 12n
FLOPs  = n additions                              = n
```

At `n` = 10 million: 120 MB moved, 10 million FLOPs.

Worked, for matrix multiply at `M = N = K = 1024`:

```
bytes  = read A (4*M*K) + read B (4*K*N) + write C (4*M*N)
       = 4 * 3 * 1024^2 = 12.6 MB
FLOPs  = 2 * M * N * K = 2.1 billion
```

**Look at those two side by side.** Vector add: 120 MB for 10 MFLOP. Matmul:
12.6 MB for 2100 MFLOP. The matrix multiply does 170x the arithmetic while
moving a tenth of the data.

---

## 3. Arithmetic intensity

The ratio of those two numbers is the single most useful quantity in
performance work:

```
arithmetic intensity = FLOPs / bytes
```

```
vector add : 10e6 / 120e6     = 0.083 FLOP/byte
SAXPY      : 2n / 12n         = 0.17  FLOP/byte
dot product: 2n / 8n          = 0.25  FLOP/byte
matmul 1024: 2.1e9 / 12.6e6   = 170   FLOP/byte
```

Three orders of magnitude between the ends of that list. **Arithmetic intensity
is a property of the algorithm, not of the hardware** — it tells you how much
work you get per byte you are forced to fetch.

Now bring in the hardware. Every machine has two peak rates:

```
peak bandwidth : bytes per second it can move
peak compute   : FLOPs per second it can do
```

Their ratio is the **ridge point** — the arithmetic intensity at which the
machine is perfectly balanced:

```
ridge = peak_compute / peak_bandwidth
```

A laptop CPU might be 100 GFLOP/s and 50 GB/s, giving a ridge of 2 FLOP/byte.
An A100 is 19,500 GFLOP/s (fp32) and 1,555 GB/s, giving 12.5. With tensor cores
it is 312,000 GFLOP/s, giving 200.

Then:

- **Intensity below the ridge → memory-bound.** The arithmetic units idle
  waiting for data. Your time is `bytes / peak_bandwidth`, and making the maths
  faster changes nothing.
- **Intensity above the ridge → compute-bound.** The memory system keeps up.
  Your time is `FLOPs / peak_compute`.

Vector add at 0.083 is memory-bound on absolutely everything, by a factor of
hundreds. Matmul at 170 is compute-bound on a CPU and only just so on an A100
with tensor cores — which is exactly why Chapter 13 is interesting.

**That is the roofline model.** Chapter 01 draws it as a graph and gives it a
name, but the content is this section.

---

## 4. Predicting a runtime

The recipe:

```
1. count bytes
2. count FLOPs
3. intensity = FLOPs / bytes
4. compare with the machine's ridge point
5. memory-bound?  time = bytes / bandwidth
   compute-bound? time = FLOPs / compute_rate
```

Vector add, `n` = 10 million, on a machine doing 50 GB/s:

```
bytes = 120 MB,  intensity 0.083, well below ridge -> memory-bound
time  = 120e6 / 50e9 = 2.4 ms
```

If you measure 2.6 ms, your model is right and there is nothing to win. If you
measure 20 ms, something is wrong and you now have a specific question to ask
rather than a vague feeling of slowness.

**Working backwards is just as useful**, and is how every gate in this course
is phrased. You measured 3 ms for 120 MB:

```
achieved bandwidth = 120e6 / 3e-3 = 40 GB/s
as a fraction of peak = 40 / 50 = 80%
```

80% of peak on a memory-bound kernel is a good result. 12% means go and find
out why.

**One honest complication: write-allocate.** On most hardware, writing to
memory you have not read first still causes a read of that cache line. So
`c[i] = a[i] + b[i]` may move 16n bytes rather than 12n. When your measured
bandwidth seems to exceed peak by about a third, this is usually why. Chapter
03 covers it properly; for now, know that your byte count is a model and models
have assumptions.

---

## 5. Why this governs what you optimise

Two consequences that run through every later chapter.

**Fusion.** Three separate element-wise passes each read and write the array:

```
b = a * 2       read a, write b
c = b + 1       read b, write c
d = c * c       read c, write d
                total: 6n * 4 bytes
```

Fuse them into one loop and you read `a` once and write `d` once: `2n * 4`
bytes, a 3x reduction in traffic for identical arithmetic. Since this is
memory-bound, 3x less traffic is 3x faster. That is Chapter 10, and `20%` of
all practical kernel wins.

**Tiling.** The naive matmul of P7.4 re-reads `A` and `B` from memory far more
than the minimum, so its *effective* intensity is much worse than the 170 we
computed. Tiling restructures the loops so that re-reads hit cache instead of
RAM, moving the real intensity back towards the theoretical one. Chapters 06
and 12.

Both are the same move: **the arithmetic is fixed, so improve the ratio by
reducing the bytes.**

---

## 6. Exercises

The first four are pencil and paper. Do them before coding.

### P8.1 - Count bytes and FLOPs

For `n` = 1 million floats unless stated:

(a) `for i: c[i] = a[i] + b[i]`
(b) `for i: a[i] = a[i] * 2.0f`  (in place — careful)
(c) `for i: sum += a[i]`
(d) `for i: sum += a[i] * b[i]`
(e) `for i: y[i] = alpha * x[i] + y[i]`
(f) Matrix multiply, `M = N = K = 512`
(g) Transpose of a 1024x1024 matrix
(h) For each: bytes, FLOPs, intensity. Rank them by intensity.

### P8.2 - Your machine's ridge point

(a) Find your CPU's memory bandwidth. (Mac: `sysctl hw.memsize` for capacity;
    for bandwidth, either look up your chip or measure it in P8.3.)
(b) Estimate peak FLOP/s: cores x clock x FLOPs per cycle per core. Look up the
    last number for your chip, or assume 16 for a modern core with AVX/NEON.
(c) Ridge point?
(d) Which of P8.1's operations are memory-bound on your machine? Any compute-
    bound?
(e) Now look up an A100: 1555 GB/s, 19.5 TFLOP/s fp32. Its ridge point? How
    does it compare with yours, and what does that difference say about which
    machine is harder to keep busy?

### P8.3 - Measure, then predict

(a) Write a loop summing 100 million floats. Time it.
(b) Compute achieved bandwidth from your byte count.
(c) That is roughly your machine's streaming bandwidth. Compare with (a) of
    P8.2.
(d) Now **predict** the time for a vector add of 100 million elements, using
    that measured bandwidth. Write the number down.
(e) Measure it. How far off were you?
(f) If you were off by more than 30%, find out why before moving on. Candidate
    explanations: write-allocate (section 4), the compiler vectorised one loop
    and not the other, or your timing included the allocation.

### P8.4 - Fusion, measured

(a) Write the three-pass version from section 5 over 50 million floats.
(b) Compute its predicted time from your measured bandwidth.
(c) Measure. 
(d) Fuse into one loop. Predict. Measure.
(e) What speedup did you get, and what did you predict? If the fused version is
    *more* than 3x faster, what else changed?
(f) Now make the operation expensive — `d = exp(sin(c))` instead of `c * c`.
    Re-measure both. Does fusion still help as much? Explain using intensity.

### P8.5 - The gates you are about to meet

Open `Kernels/README.md` and look at the progress checklist.

(a) `02-vector-add` requires >= 80% of the copy ceiling. Given section 4, what
    does that mean in terms of bytes and seconds?
(b) `09-histogram` requires only 50%. Why would a histogram be allowed a lower
    bar than a vector add? (Guess. You will find out in Chapter 09.)
(c) `11-gemm-ladder` is measured against cuBLAS rather than against a bandwidth
    ceiling. Why is a percentage of peak bandwidth the wrong bar for a matrix
    multiply?
(d) `21-int4-gemv` is gated on bandwidth even though it is a matrix operation.
    Given that its weights are 4 bits, estimate its arithmetic intensity and
    explain the choice.

---

## What you should now believe

- Every loop has a byte count and a FLOP count, and you can work out both
  without running anything.
- Their ratio is arithmetic intensity, a property of the algorithm.
- The machine has a ridge point; below it you are memory-bound and above it you
  are compute-bound, and that decides what is worth optimising.
- Predicted time is `bytes / bandwidth` or `FLOPs / compute_rate`, whichever
  side you are on.
- Measured time converts back into achieved bandwidth, and "percent of peak" is
  how you know whether you are finished.
- Fusion and tiling both work by reducing bytes while holding arithmetic fixed.

---

## You are ready

That was the last prerequisite chapter. You can now:

- read and write pointers, and know why kernel outputs are pointers
- flatten and unflatten multi-dimensional indices without thinking
- say why one loop order is ten times faster than another
- explain why a GPU reduction is not bit-reproducible
- read the C++ in `bench.cuh`
- diagnose a compiler error and reach for a sanitizer
- recognise a reduction, a matmul, a softmax and a normalisation on sight
- predict a loop's runtime and check your prediction

Do the six exercises in [`exercises/`](exercises) if you have not. Then start
the real course at [Chapter 00 - Why GPUs exist](../chapters/00-why-gpus-exist.md),
where the first thing you will read is that a GPU is a machine for hiding the
latency you measured in P0.4.
