# Chapter 11 - GEMM I: from naive to tiled

> Matrix multiply is the most optimised routine in computing. It is also the
> best teacher, because the path from the obvious version to a fast one
> traverses every idea in this course: coalescing, shared memory, registers,
> ILP, occupancy. Climb it once, properly, and everything else is a variation.

---

## 1. The target

```
C = A @ B,   A is M x K,  B is K x N,  C is M x N
C[i][j] = sum_k A[i][k] * B[k][j]
```

```
W = 2 M N K FLOPs
Q_compulsory = 4(MK + KN + MN) bytes
I = 2MNK / (4(MK + KN + MN))
```

For M = N = K = 4096: `I = N/6 = 683` FLOPs per byte, against an fp32 knee
around 10. **Deeply compute-bound, in principle.** The predicted time on an
A100 is `2*4096^3 / 19.5e12 = 7.0 ms`.

That "in principle" is carrying everything. `I = 683` assumes you read each
element of A and B exactly once. The naive kernel reads each one 4096 times.

---

## 2. Why the naive version is 40x off

```cuda
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float acc = 0;
        for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
        C[i * N + j] = acc;
    }
}
```

Correct. Now count traffic. Each thread does `K` iterations, reading 2 floats
each:

```
Q = 2 * 4 * M * N * K = 5.5e11 bytes  for 4096^3
I = 1.37e11 / 5.5e11 = 0.25 FLOPs/byte
T = 5.5e11 / 2.0e12 = 275 ms
```

**39x slower than the compute roof, and firmly memory-bound.** The FLOP count
is identical to the optimal version. Only the traffic differs.

The L1/L2 caches rescue some of this in practice -- the `B` access is shared
across threads in a warp -- so you will measure something better than 275 ms.
But a cache is a hardware guess. The whole optimisation program of this chapter
is replacing that guess with a plan.

### One free fix first

In the kernel above, which index should map to `threadIdx.x`? With `j` (the
column of B and C) as the fast index, a warp reads `B[k*N + j..j+31]`:
consecutive. Coalesced. Swap `i` and `j` and every access becomes stride-`N`.
**Same code, 8x difference.** Chapter 03, still earning.

---

## 3. The optimisation ladder

You climb this in `11-gemm-ladder`. Here is the *shape* of each step and the
question it answers. Not the code.

### v1: naive

One thread per output element. Baseline. Measure it; everything else is
relative to this.

### v2: shared-memory tiling

Each block computes a `BM x BN` tile of C by looping over K in chunks of `BK`:

```
for kt in 0..K/BK:
    cooperatively load A[BM x BK] and B[BK x BN] into shared memory
    __syncthreads()
    each thread accumulates its partial dot products from shared memory
    __syncthreads()
```

**The reuse question:** with a 32x32 tile, how many threads use each element of
A loaded into shared memory? That number divides your traffic. Work out the new
`I`, then predict the speedup before you measure it.

With BM = BN = BK = 32:

```
Q = 2 * 4 * M * N * K / 32 = 1.7e10
I = 8 FLOPs/byte
```

8 against a knee of 10: still slightly memory-bound, and roughly 32x better
than naive. Good, and not enough.

### v3: 1D register tiling (thread coarsening)

Each thread computes **several** outputs -- say 8 in a column -- instead of
one.

**The question:** when a thread computes 8 outputs sharing the same `B[k][j]`
value, how many times does it read that value from shared memory? Once, into a
register, and reuses it 8 times. So:

- Shared-memory traffic per FLOP drops 8x.
- Arithmetic intensity against *global* memory rises because tiles get bigger.
- Each thread now has 8 accumulators live: more registers, more ILP, lower
  occupancy.

That last line is Chapter 04's trade, and this is where it pays. Expect
roughly 2-3x over v2.

### v4: 2D register tiling

Each thread computes a `TM x TN` block of outputs, typically 8x8 = 64.

The inner loop becomes an **outer product**: load `TM` values of A and `TN`
values of B into registers, then do `TM * TN` FMAs.

```
registers for A fragment : TM
registers for B fragment : TN
registers for accumulators: TM * TN
FMAs per shared-memory load: (TM * TN) / (TM + TN)
```

For 8x8: 64 FMAs per 16 shared loads = **4 FMAs per shared load**, versus 0.5
for v2. That ratio is the entire optimisation. Count it for your tile sizes
before you write the kernel.

Cost: 64 + 16 = 80 registers minimum, plus addressing. At 255 per thread you
fit, but occupancy drops to 25-30%. Verify that is still enough to cover
latency -- it will be, because you now have enormous ILP.

Expect 2-3x over v3, putting you in the region of 50-70% of cuBLAS.

### v5: vectorized loads and layout

- **`float4` global loads** for the tile loads. Chapter 03: fewer instructions,
  more bytes in flight.
- **Transpose the A tile in shared memory** as you store it, so the inner loop
  reads A column-wise without bank conflicts and with `float4` accesses.
- **Resolve remaining bank conflicts** by padding or swizzling. Check with the
  profiler, not by reasoning alone.
- **Tune `BM, BN, BK, TM, TN`.** The search space is large and the optimum is
  machine-specific. This is why cuBLAS ships hundreds of kernels and picks at
  runtime.

Target: 70%+ of cuBLAS. That is the gate.

---

## 4. The hierarchy, named

What you have built by v4 is the standard structure every fast GEMM uses:

```
GLOBAL   -> SHARED      block tile     BM x BN x BK    (cooperative load)
SHARED   -> REGISTERS   warp/thread    TM x TN         (outer product)
REGISTERS-> FMA         instruction    1 x 1 (or 16x8x16 on tensor cores)
```

Three levels of blocking, one per level of the memory hierarchy. CUTLASS calls
them threadblock tile, warp tile, and instruction tile, and its documentation
is worth reading once you have built this yourself -- it will read as a
description of what you did, rather than as a foreign language.

---

## 5. Choosing tile sizes

Four constraints to satisfy simultaneously:

**Shared memory.** `(BM*BK + BK*BN) * 4` bytes per block, doubled if you double
buffer. For 128x128x8 that is 8 KB; with 164 KB per SM you could hold many
blocks. For 128x128x32 it is 32 KB and you get 5 blocks per SM at best.

**Registers.** `TM*TN + TM + TN` plus addressing, roughly. For 8x8 that is
~90-110 in practice. Check with `-Xptxas -v`, and **check for spills** -- an
8x8 tile that spills is slower than a 4x4 one that does not.

**Threads.** `(BM*BN) / (TM*TN)` threads per block. 128x128 tiles with 8x8
thread tiles = 256 threads. Convenient, which is why those numbers are so
common.

**Loads per thread.** Each thread must load `(BM*BK + BK*BN) / threads`
elements per iteration. If that is not a whole number, or not a multiple of 4
for `float4`, your loading code gets ugly and slow. Choose sizes that divide
cleanly.

Common configurations worth knowing: 128x128x8 with 8x8 thread tiles (256
threads), 128x64x16, 64x64x16 with 4x4 tiles. Start at 128x128x8.

---

## 6. Correctness traps

**Bounds.** Real matrices are not multiples of 128. Options: guard every access
(costly in the inner loop), pad the shared tile with zeros at the edges (clean,
and zeros contribute nothing to a dot product), or write a separate tail
kernel. Padding with zeros is the standard answer -- understand why it is safe.

**The two barriers.** One after loading the tile, one after computing with it
and before overwriting it in the next iteration. Missing the second is the
classic race.

**Accumulate in fp32** even for fp16 inputs. Always. Chapter 15.

**Verify against cuBLAS**, with a tolerance. A GEMM with K = 4096 sums 4096
products; fp32 accumulation gives a relative error around `sqrt(K) * eps ~
1e-5`. A tolerance of 1e-3 is reasonable; demanding bit-exactness against
cuBLAS is meaningless since it uses a different accumulation order.

---

## 7. Common failure modes

**v2 barely faster than v1.** Check the reuse actually happened: is the inner
loop reading shared memory or global? Is the tile load coalesced?

**v3/v4 slower than v2.** Spilling. `-Xptxas -v`. Or the compiler failed to
unroll the inner loops, so your accumulators became a dynamically-indexed local
array (Chapter 04 section 4). `#pragma unroll` with compile-time bounds.

**Stuck at ~40% of cuBLAS.** Usually bank conflicts in the shared-memory reads
of the inner loop. Measure them.

**Correct for 4096, wrong for 4000.** Boundary handling.

**Huge variance between runs.** Clock throttling, or you forgot to warm up.

**Faster than cuBLAS.** Check your correctness tolerance and your FLOP count,
in that order. It happens for specific shapes -- cuBLAS is tuned for common
ones -- but verify before celebrating.

---

## 8. Exercises

### 11.1 - The traffic ladder

For M = N = K = 4096 fp32, compute `Q`, `I` and the predicted time on your GPU
for:

(a) naive
(b) 32x32x32 shared tiling
(c) 64x64x16 tiling, 4x4 thread tiles
(d) 128x128x8 tiling, 8x8 thread tiles
(e) the compulsory lower bound

Plot `I` for each on your roofline from Chapter 04 exercise 4.5. Mark where
each crosses from memory-bound to compute-bound. Which is the first version
that is genuinely compute-bound?

### 11.2 - The FMA-to-load ratio

For a `TM x TN` thread tile:

(a) Derive FMAs per shared-memory load as a function of TM and TN.
(b) Evaluate for 1x1, 2x2, 4x4, 8x8, 16x16.
(c) Registers needed for each.
(d) At what point do you exceed 255 registers per thread?
(e) Occupancy at each, on your GPU.
(f) Plot (predicted throughput) against TM for TM = TN. There is an optimum
    and it is not the largest tile. Where is it, and what makes it turn over?

### 11.3 - Shared memory budget

(a) Shared bytes for 128x128x8 tiles, fp32.
(b) Blocks per SM from the shared limit alone, on your GPU.
(c) Same with double buffering.
(d) Same for BK = 32.
(e) Which (BM, BN, BK) maximises blocks per SM while keeping >= 4 FMAs per
    shared load? Is that the configuration you would ship?

### 11.4 - Design before you build

Write the full design for your v4 before writing code:

```
BM, BN, BK      = ______
TM, TN          = ______
threads/block   = ______
shared bytes    = ______
registers/thread (estimate) = ______
occupancy       = ______
elements each thread loads per K-step = ______
can they be float4? ______
FMAs per shared load = ______
predicted TFLOP/s = ______
predicted % of cuBLAS = ______
```

Then build it and compare every line against reality. The lines that are wrong
are your learning.

### 11.5 - Index arithmetic on paper

For 128x128x8 tiles, 8x8 thread tiles, 256 threads:

(a) Which 64 elements of C does thread `t` own? Write the formula.
(b) Which elements of the A tile and B tile does thread `t` read in inner-loop
    step `k`?
(c) For loading: each thread loads how many elements of A? Of B? Give the
    mapping from `threadIdx.x` to the element, such that the global read is
    coalesced and `float4`-able.
(d) Trace warp 0's 32 global addresses in the A load. Sectors?
(e) Trace warp 0's 32 shared addresses in the inner-loop A read. Bank
    conflicts? If yes, what layout change fixes it?

This exercise is tedious and it is the difference between a kernel that works
and three days of debugging. Do it on paper.

---

## Exercises to run

- [`11-gemm-ladder`](../exercises/11-gemm-ladder) - five versions, gated at
  70% of cuBLAS fp32 for v5.

---

## What you should now believe

- GEMM's FLOP count is fixed; everything is about traffic.
- Tiling in shared memory divides global traffic by the tile size.
- Register tiling divides *shared* traffic, and the FMA-per-shared-load ratio
  is the number to maximise.
- Three levels of blocking map to three levels of memory hierarchy.
- Low occupancy with high ILP is the correct operating point here.
- Tile-size selection is a constrained optimisation, which is why real
  libraries autotune.

Next: [Chapter 12 - GEMM II](12-gemm-ii.md).
