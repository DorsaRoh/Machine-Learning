# Chapter 01 - The cost model: roofline

> The skill that separates a kernel engineer from a person who writes CUDA is
> the ability to say, before compiling, "this should take 1.4 milliseconds",
> and then be right. This chapter teaches that, and it is the most valuable
> thing in the course.

---

## 1. Two numbers describe any kernel

For any computation, count two things:

- **W** = work, in FLOPs.
- **Q** = traffic, in bytes moved between the GPU and its global memory.

Their ratio is the **arithmetic intensity**:

```
I = W / Q        [FLOPs per byte]
```

The machine also has two numbers:

- **P** = peak compute, in FLOP/s.
- **B** = peak bandwidth, in bytes/s.

The achievable performance is then

```
attainable FLOP/s = min(P, B x I)
```

and the predicted runtime is

```
T = max(W / P, Q / B)
```

That is the roofline model. The whole thing. Two counts, two constants, a
`min`. It is crude, it ignores caches, latency, occupancy and instruction mix,
and it will still be the most accurate predictive tool you own.

```
  attainable FLOP/s (log)
    |
  P |................________________________  <- compute roof
    |              ./
    |            ./   slope = B
    |          ./
    |        ./
    |      ./
    +----------------------------------------- arithmetic intensity (log)
             ^
          I* = P/B  (the "knee")

   I < I* : memory-bound. Runtime is Q/B. Optimise bytes.
   I > I* : compute-bound. Runtime is W/P. Optimise math.
```

The knee `I* = P/B` is the single most important property of a GPU. For an
A100: 19.5 TFLOP/s / 2.0 TB/s = **~10 FLOPs per byte** in fp32. With fp16
tensor cores: 312 / 2.0 = **~156**. On an H100: **~295**.

---

## 2. Counting Q honestly

Counting W is usually easy. Counting Q is where people deceive themselves.

**Rule: Q is the compulsory traffic between HBM and the chip, assuming perfect
caching.** Count each distinct byte your algorithm *must* read at least once,
and each byte it must write.

Vector add, N elements, fp32:

```
W = N FLOPs
Q = 4N (read a) + 4N (read b) + 4N (write c) = 12N bytes
I = N / 12N = 0.083 FLOPs/byte
```

Three subtleties that change the number:

**(a) Write-allocate.** On most GPUs, writing to a location you have not read
first still causes the containing cache line to be *fetched* from memory, then
modified, then written back. A pure write of 4N bytes can cost 8N bytes of
traffic. This is why `dram__bytes` in the profiler often exceeds your
hand-count by exactly the size of your output. Streaming stores exist to avoid
it (Chapter 03).

**(b) Reuse you did not get.** If your algorithm reads `a[i]` ten times and the
cache holds it, Q counts it once. If your access pattern is such that it is
evicted between uses, Q counts it ten times. **The gap between the compulsory Q
and the actual Q is where most optimisation lives.** Tiling (Chapter 06) is
precisely the art of turning actual Q back into compulsory Q.

**(c) Granularity waste.** Memory moves in 32-byte sectors. If a warp reads 32
floats that are 128 bytes apart, it requests 128 bytes of useful data and the
memory system moves 32 x 32 = 1024 bytes. Q is 8x your hand-count. This is the
coalescing problem and it is Chapter 03.

So really there are three Q's, and knowing which you mean is half of
performance engineering:

```
Q_compulsory  what the algorithm must move
Q_actual      what the algorithm as written moves (reuse failures)
Q_hardware    what the memory system actually moves (granularity waste)
```

Your job in any memory-bound kernel: drive `Q_hardware` down to
`Q_compulsory`. When they are equal, you are done, and no amount of further
cleverness will help.

---

## 3. Worked predictions

### Example 1: SAXPY

`y[i] = a * x[i] + y[i]`, N = 64M, fp32, on a 2 TB/s A100.

```
W = 2N = 1.34e8 FLOPs
Q = 4N (read x) + 4N (read y) + 4N (write y) = 8.05e8 bytes
I = 0.167
T_mem     = 8.05e8 / 2.0e12 = 403 us
T_compute = 1.34e8 / 19.5e12 = 7 us
T = max = 403 us, memory-bound by 60x.
```

Prediction: **403 microseconds, achieving 0.33 TFLOP/s**, which is 1.7% of
peak. If you measure 420 us, your kernel is essentially perfect. Chasing the
remaining 4% is not optimisation, it is noise.

### Example 2: GEMM

`C = A @ B`, all N x N, fp32, N = 4096.

```
W = 2 N^3 = 1.37e11 FLOPs
Q_compulsory = 3 * 4 * N^2 = 2.0e8 bytes   (read A, read B, write C once each)
I = 2N^3 / (12 N^2) = N/6 = 683 FLOPs/byte
```

683 is far above the fp32 knee of 10, so compute-bound:

```
T = 1.37e11 / 19.5e12 = 7.0 ms
```

But that assumes `Q = Q_compulsory`, which assumes you can reuse every element
of A and B from on-chip memory. A naive kernel where each thread computes one
output by looping over K reads 4096 elements of A and 4096 of B per output:

```
Q_naive = 2 * 4 * N^3 = 5.5e11 bytes
I_naive = 1.37e11 / 5.5e11 = 0.25 FLOPs/byte     (!!)
T_naive = 5.5e11 / 2.0e12 = 275 ms
```

**39x slower, and now memory-bound.** The entire content of Chapter 11 is
clawing I from 0.25 back up toward 683 by means of on-chip reuse. Notice that
the FLOP count never changes. Only the traffic does.

### Example 3: Softmax

One row of 1024 fp32, 4096 rows.

```
W: max (1024 comparisons) + 1024 exp + 1024 adds + 1024 divides ~ 4e3 "FLOPs"
Q_compulsory = 4 * 1024 read + 4 * 1024 write = 8192 bytes per row
I ~ 0.5
```

Memory-bound, hard. So the *only* thing that matters for softmax performance is
how many times you read the row. The textbook three-pass version (find max,
sum exp, normalise) reads the row three times if it does not fit in registers
or shared memory:

```
Q_3pass = 3 reads + 1 write = 16 KB/row
Q_1pass = 1 read  + 1 write = 8 KB/row     -> 2x faster
```

That factor of two is entirely algorithmic, costs no extra math, and is the
online softmax trick of Chapter 10. It is also, scaled up, FlashAttention.

### Example 4: LLM decode

Generating one token from a 7B-parameter model in fp16, batch size 1.

```
W = 2 * 7e9 = 1.4e10 FLOPs
Q = 2 bytes * 7e9 params = 1.4e10 bytes   (every weight read exactly once)
I = 1.0 FLOP per byte
```

Intensity of **1**, against a tensor-core knee of 156. This is memory-bound by
a factor of 156, which means:

```
T = 1.4e10 / 2.0e12 = 7 ms per token  ->  ~143 tokens/s, maximum, ever
```

and the tensor cores are at **0.6% utilisation**. You now understand, from
first principles and in two lines of arithmetic, why:

- LLM inference is priced by memory bandwidth, not FLOPs.
- Quantizing to int4 gives a ~4x speedup almost for free (Chapter 21): it cuts
  Q by 4 and the kernel is 100% Q-bound.
- Batching is nearly free until you hit the knee: batch 128 raises I to ~128,
  still under 156, so 128 users cost barely more than 1.
- Speculative decoding works: it converts a latency problem into a throughput
  problem the machine was always able to absorb.

Four lines of roofline arithmetic explain the economics of an industry. This is
why the chapter exists.

---

## 4. The third regime: latency-bound

The roofline has two roofs, but there is a third way to be slow that it does
not show: **you are under both roofs because there is not enough work in
flight**.

Symptoms: memory throughput at 20% of peak, compute at 10% of peak, nothing
saturated, kernel still slow. The profiler section `SpeedOfLight` showing both
numbers low is the tell.

Causes, in rough order of frequency:

1. Grid too small -- you are not using all the SMs.
2. Long dependency chains inside each thread, so there is only ever one
   instruction in flight per thread.
3. Occupancy too low to cover memory latency (back to Little's Law).
4. Synchronisation -- every `__syncthreads()` drains the pipeline.
5. The kernel is so short that launch overhead dominates.

The fix is always "more independent work in flight", via more threads, more
work per thread done independently, or deeper unrolling.

---

## 5. Where the model lies to you

Use it knowing the failure modes:

- **It ignores caches.** A kernel re-reading a 1 MB table 100 times has
  `Q_compulsory = 1 MB`, and if it fits in L2 your measurement will blow past
  the "roof". The roofline then is a floor on performance, not a ceiling. When
  benchmarking, decide deliberately whether you want the cached or uncached
  number -- `bench.cuh` flushes L2 between iterations so you get the honest,
  uncached one.
- **It assumes perfect overlap** of compute and memory. Real kernels have
  phases; a kernel that loads everything then computes everything achieves
  neither roof.
- **It treats all FLOPs equally.** A `__expf` is not an FMA. Tensor-core FLOPs
  and FP32 FLOPs are different roofs on the same chip; draw both.
- **Peak numbers are unreachable.** 85-90% of datasheet bandwidth is excellent;
  95% of peak FLOPs is cuBLAS-grade. Set your expectations against *measured*
  ceilings, which is why `bench::copy_ceiling_gbs()` measures rather than
  looks up.

---

## 6. The habit to build

Before writing any kernel, fill this in:

```
1. W (FLOPs)                 = ?
2. Q_compulsory (bytes)      = ?
3. I = W/Q                   = ?
4. I* for this GPU & dtype   = ?
5. Bound by                  = memory / compute
6. Predicted T               = ?
7. Predicted achieved GB/s or TFLOP/s = ?
```

Then write it, measure, and compare. When reality disagrees with the
prediction, one of these is true and all three are worth knowing:

- Your traffic count was wrong (usually: reuse you assumed you would get, and
  did not; or granularity waste).
- Your kernel is latency-bound (not enough in flight).
- The model's assumptions broke (caching, phase separation, instruction mix).

Diagnosing *which* is the skill. By Chapter 16 you will do it from counters in
under a minute.

---

## 7. Exercises

### 1.1 - Derive the GEMM intensity

Show that for `C = A @ B` with A being M x K, B being K x N, all fp32, read
once each:

```
I = 2MNK / (4(MK + KN + MN))
```

(a) Simplify for M = N = K.
(b) For M = N = 4096, what does I become as K shrinks to 1? What operation is
that, and what is it bound by?
(c) At what M (with N = K = 4096) does the problem cross your GPU's fp32 knee?
Its fp16 tensor-core knee? This number is why inference servers batch requests.

### 1.2 - Predict before you measure

For each, write down the predicted runtime on *your* GPU before you touch a
keyboard. You will verify several of these in later chapters, so date your
predictions.

(a) Copy 1 GB device-to-device.
(b) Sum 256M fp32 values to a scalar.
(c) Transpose a 8192 x 8192 fp32 matrix.
(d) LayerNorm over 8192 rows of 4096 fp32.
(e) `C = A @ B` for 1024 x 1024 fp32, ideal reuse.
(f) The same GEMM with no reuse at all (naive).
(g) Attention: batch 8, 16 heads, sequence 2048, head dim 64, fp16, computing
    and materialising the full 2048 x 2048 score matrix.
(h) The same attention, never materialising scores (tiled). Compare Q.

### 1.3 - The intensity of fusion

You have three elementwise kernels in a row, each reading N floats and writing
N floats: `y = relu(x)`, `z = y * 2`, `w = z + 1`.

(a) Total Q unfused.
(b) Total Q if fused into one kernel.
(c) Speedup.
(d) Now the chain is 10 ops long. Speedup?
(e) What is the limit as the chain grows, and what does that tell you about why
`torch.compile` exists?

### 1.4 - Find the knee of your machine

Design (on paper) a microbenchmark whose arithmetic intensity you can sweep
continuously from 0.1 to 1000 FLOPs/byte, so that plotting achieved FLOP/s
against I traces out your GPU's actual roofline.

(a) What does the kernel look like?
(b) How do you stop the compiler from deleting math whose result you do not
    use? (This matters enormously and catches everyone once.)
(c) How do you prevent the L2 cache from flattening the memory side?
(d) Sketch the plot you expect. Where is the knee? Is the measured knee to the
    left or right of `P/B`, and why?

You will build this in Chapter 04. Design it first.

### 1.5 - Interrogate a real result

A colleague reports: "my elementwise kernel hits 4.2 TB/s on an A100."

(a) The datasheet says 2.0 TB/s. List every way this number could be
    technically true.
(b) What single change to their benchmark would produce the honest figure?
(c) They then report 1.95 TB/s and want to optimise further. What do you tell
    them?

---

## Exercise: `00-hello-gpu`

Now run something. [`exercises/00-hello-gpu`](../exercises/00-hello-gpu)
probes your GPU, measures its real streaming bandwidth, and makes you compute
its roofline knee from the numbers it prints.

```bash
cd Kernels/exercises/00-hello-gpu && make run
```

---

## What you should now believe

- Runtime is `max(W/P, Q/B)` until proven otherwise.
- The number that decides a kernel's fate is arithmetic intensity.
- Counting bytes honestly -- compulsory vs actual vs hardware -- is the core
  analytical skill of this field.
- Optimisation of a memory-bound kernel means reducing Q. Nothing else works.
- Low on both roofs means latency-bound, and the fix is more work in flight.

Next: [Chapter 02 - Your first kernel](02-first-kernel.md).
