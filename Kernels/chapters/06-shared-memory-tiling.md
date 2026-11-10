# Chapter 06 - Shared memory and tiling

> The first genuinely powerful idea in the course. Shared memory is how you
> turn an access pattern the memory system hates into one it loves, and how you
> turn repeated global reads into a single read plus on-chip reuse. Everything
> from here to FlashAttention is this idea at different scales.

---

## 1. What shared memory is

A software-managed scratchpad, physically on the SM, shared by all threads of
one block.

```cuda
__global__ void k(...) {
    __shared__ float tile[32][32];   // static: size known at compile time
    ...
}

extern __shared__ float buf[];       // dynamic: size given at launch
k<<<grid, block, bytes>>>(...);
```

| | shared memory | global memory |
|---|---|---|
| Latency | ~25 cycles | ~500 cycles |
| Bandwidth | ~20 TB/s aggregate | 1-3 TB/s |
| Scope | one block | everything |
| Lifetime | the block | the allocation |
| Managed by | you | the hardware |

That last row is the point. L1 is a cache: it decides what to keep. Shared
memory is a scratchpad: *you* decide. When you know your access pattern better
than a hardware heuristic does -- and in a kernel you wrote, you always do --
explicit beats automatic.

On most architectures L1 and shared memory are the *same physical SRAM*, split
by a configurable carveout. Asking for more shared memory leaves less L1.

---

## 2. The two things it is for

### Use 1: changing the access pattern

The transpose problem from Chapter 03: reads want row-major, writes want
column-major, and no indexing fixes both. With a staging area the conflict
disappears:

```
1. read a 32x32 tile from global, coalesced (rows)
2. __syncthreads()
3. write it to global, coalesced (also rows -- of the output)
   reading it out of shared memory transposed
```

The transpose still happens. It happens **in shared memory**, where strided
access costs ~25 cycles instead of ~500 and no DRAM sectors are wasted. Both
global accesses are now perfectly coalesced.

```cuda
__shared__ float tile[32][32];
// coalesced load: lane i reads in[y*n + x0+i]
tile[threadIdx.y][threadIdx.x] = in[y * n + x];
__syncthreads();
// coalesced store: lane i writes out[y2*n + x20+i], reading tile transposed
out[y2 * n + x2] = tile[threadIdx.x][threadIdx.y];
```

Notice the asymmetry: the write to `tile` is by `[y][x]` and the read is by
`[x][y]`. One of those two shared accesses is going to hit bank conflicts.
Section 4.

### Use 2: reuse

If a value is read by many threads in a block, read it from global **once** and
share it:

```
without tiling: each of P threads reads a value  -> P global reads
with tiling:    one thread reads it to shared    -> 1 global read
                                                    + P shared reads
```

This divides `Q` by the reuse factor, which multiplies arithmetic intensity by
the same factor, which moves you right along the roofline. For a GEMM with a
32x32 tile, each element of A is used by 32 threads: Q drops 32x and the kernel
goes from hopelessly memory-bound to nearly compute-bound. That is Chapter 11,
and it is the single largest optimisation in this course.

**The tiling pattern, in general:**

```
for each tile of the problem:
    cooperatively load the tile into shared memory   (coalesced)
    __syncthreads()
    every thread computes using the shared tile      (reuse)
    __syncthreads()
```

Both barriers are mandatory. The first stops a thread reading data that has not
been written yet. The second stops a thread overwriting the tile while others
are still reading the previous iteration's data. Missing the second one is the
single most common CUDA race, it produces intermittent wrong answers, and
`compute-sanitizer --tool racecheck` finds it instantly.

---

## 3. Banks

Shared memory is divided into **32 banks**, each 4 bytes wide, interleaved:

```
address (floats):  0   1   2   3  ...  31   32   33  ...
bank:              0   1   2   3  ...  31    0    1  ...
```

so `bank = (byte_address / 4) % 32`.

The hardware services a warp's shared access in one cycle **if each bank is
touched by at most one lane**. If k lanes hit different addresses in the same
bank, the access serialises into k cycles: a **k-way bank conflict**.

The exception that matters: **if several lanes read the *same* address, it is a
broadcast and costs nothing.** Same bank, same address: free. Same bank,
different addresses: serialised.

### Examples

```cuda
__shared__ float s[32][32];

// lane i reads s[0][i]: banks 0..31, one each. CONFLICT-FREE.
float a = s[0][threadIdx.x];

// lane i reads s[i][0]: address 32*i floats -> bank (32*i)%32 = 0 for ALL i.
// 32-WAY CONFLICT. 32x slower.
float b = s[threadIdx.x][0];

// lane i reads s[0][2*i]: banks 0,2,4,...,62 mod 32 -> each bank twice.
// 2-WAY CONFLICT.
float c = s[0][2 * threadIdx.x];

// every lane reads s[0][0]: BROADCAST. Free.
float d = s[0][0];
```

The column access is the killer, and column access is exactly what a transpose
does.

### Fix 1: padding

```cuda
__shared__ float s[32][33];   // one extra column
```

Now row `i` starts at float offset `33*i`, so `s[i][0]` is in bank
`(33*i) % 32 = i`. Every lane lands in a different bank. **Conflict-free, at
the cost of 3% more shared memory.**

The `+1` padding trick is one of the highest value-per-character edits in GPU
programming. Understand *why* it works -- offsetting the row stride by one bank
-- rather than memorising it, because for `float4` or `half` data the right
padding is different and the reasoning is what transfers.

### Fix 2: swizzling

Padding wastes memory and breaks vectorized (128-bit) shared accesses, which
matters for tensor-core pipelines. The general alternative is to permute
addresses with XOR:

```cuda
int col = threadIdx.x ^ (threadIdx.y & 7);   // XOR swizzle
tile[threadIdx.y][col] = value;
```

XOR is a bijection, so no data is lost, and it scatters each row's elements
across banks differently. Every serious GEMM and attention kernel uses a
swizzle rather than padding. You will meet them again in Chapters 13 and 14.

---

## 4. The transpose, properly

Working through the conflict analysis of the tiled transpose, because it is the
canonical exercise and because the reasoning generalises.

With `__shared__ float tile[32][32]` and a 32x8 block:

**Write phase**, `tile[threadIdx.y][threadIdx.x] = in[...]`. Lane `i` of a warp
has `threadIdx.x = i`, `threadIdx.y` fixed. Address = `32*ty + i` floats, so
bank = `i`. All different. **Conflict-free.**

**Read phase**, `out[...] = tile[threadIdx.x][threadIdx.y]`. Lane `i` has
`threadIdx.x = i`, so address = `32*i + ty` floats, bank = `(32*i + ty) % 32 =
ty`. **All 32 lanes hit the same bank, different addresses. 32-way conflict.**

So the naive tiled transpose replaces a global-memory disaster with a
shared-memory disaster. It is still faster -- a 32-way shared conflict is
cheaper than 32 DRAM sectors -- but it leaves most of the win on the table.
Padding to `tile[32][33]` makes the read phase conflict-free and typically buys
another 1.5-2x.

This is the entire content of exercise `06-transpose`, and you should measure
all three versions: naive global, tiled unpadded, tiled padded.

---

## 5. Dynamic shared memory and the opt-in limit

```cuda
extern __shared__ char smem[];
float* a = reinterpret_cast<float*>(smem);
float* b = a + size_a;                 // manual sub-allocation
kernel<<<grid, block, (size_a + size_b) * sizeof(float)>>>(...);
```

Use it when the tile size is a runtime parameter. The cost is that the compiler
no longer knows the size, so it cannot do some layout optimisations, and you
must do the pointer arithmetic yourself.

The default static limit is 48 KB per block even on GPUs with 164-228 KB per
SM. To exceed it you must opt in explicitly:

```cuda
cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                     100 * 1024);
```

Forgetting this is why a kernel that should use a 96 KB tile fails at launch
with `invalid argument`. FlashAttention kernels live right at this limit, so
you will meet it for real in Chapter 19.

---

## 6. The cost of a barrier

Every `__syncthreads()` makes every warp in the block wait for the slowest.
During the wait, that SM may have nothing to issue (if no other block is
resident). In a tight loop, barriers routinely become the top stall reason.

Mitigations:

- **Fewer, bigger tiles**: more work per barrier.
- **More blocks per SM**: while one block waits, another issues. This is one of
  the few genuine arguments for higher occupancy.
- **Double buffering**: load tile `k+1` into a second buffer while computing on
  tile `k`, so there is no point where everyone waits with nothing to do. This
  is Chapter 12, and on Ampere+ it combines with `cp.async` (Chapter 14) into
  the modern pipelined GEMM.
- **Warp-level instead of block-level**: if cooperation only needs 32 threads,
  shuffles need no barrier at all.

---

## 7. Common failure modes

**Intermittently wrong results.** A missing barrier. Almost always the *second*
one, before overwriting a tile in the next loop iteration. `racecheck`.

**Hangs.** `__syncthreads()` inside divergent control flow (Chapter 05).

**Tiled version barely faster than naive.** Bank conflicts. Check
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`; it should be ~0.

**Launch fails with `invalid argument`.** Shared request exceeds 48 KB without
the opt-in, or exceeds the hardware limit entirely.

**Occupancy collapsed after adding shared memory.** The shared-memory limiter
now binds. `smem_per_SM / smem_per_block` blocks per SM -- compute it.

**Padding made it slower.** You broke `float4` alignment of shared accesses, or
you pushed shared usage over a granularity step and lost a block per SM.
Measure, do not assume.

**Correct on small inputs, wrong on large.** Tile boundary handling when the
matrix size is not a multiple of the tile size.

---

## 8. Exercises

### 6.1 - Bank conflict analysis

For `__shared__ float s[32][32]` and a full warp, give the conflict degree:

(a) `s[0][tid]`
(b) `s[tid][0]`
(c) `s[tid][tid]`
(d) `s[tid/2][tid%2]`
(e) `s[0][tid*2]`
(f) `s[0][tid*3]`
(g) `s[0][tid*32]`
(h) `s[0][0]`
(i) `s[tid%8][tid/8]`

Then redo (b), (c) and (f) for `__shared__ float s[32][33]`. Which are fixed,
and is any made *worse*?

For (f): stride 3 is odd. Does an odd stride ever conflict? State the general
rule for stride `k` and prove it.

### 6.2 - Pad a different type

`__shared__ double s[32][32]` (8 bytes per element).

(a) Which bank does `s[i][j]` live in? Careful: banks are 4 bytes wide and a
    double spans two.
(b) Does `s[tid][0]` conflict? How badly?
(c) Does `+1` padding fix it? What padding does?
(d) Same questions for `__shared__ half s[32][64]` (2 bytes).
(e) Same for `__shared__ float4 s[32][8]`.

The `+1` reflex fails for at least one of these. Knowing which, and why, is the
point.

### 6.3 - Design the swizzle

You want a 32x32 float tile with no padding (so rows stay 128-byte aligned for
`float4` access) and no bank conflicts on either row or column access.

(a) Propose an XOR swizzle `col' = col ^ f(row)`.
(b) Prove it is a bijection on each row (no data lost).
(c) Show column access `s[i][c]` for fixed `c` hits 32 distinct banks.
(d) Does row access stay conflict-free?
(e) What does the index arithmetic cost, in instructions, per access?

This is the real technique used in CUTLASS. Deriving it yourself once makes
Chapter 13 readable.

### 6.4 - Transpose budget, revisited

For the 8192 x 8192 transpose:

(a) Predict the tiled unpadded time, including the 32-way read conflict.
(b) Predict the padded time.
(c) What fraction of the copy ceiling do you expect from (b)?
(d) The tile is 32x32. Would 64x64 be better? Compute its shared footprint and
    the resulting occupancy before answering.
(e) With a 32x32 tile and 32x8 threads, each thread moves 4 elements. Why not
    use 32x32 threads so each moves 1? Give two reasons.

### 6.5 - Reuse arithmetic

A block of 16x16 threads computes a 16x16 tile of `C = A @ B`, looping over K
in steps of 16.

(a) Without shared memory, global reads per output element?
(b) With tiling, global reads per output element?
(c) Reuse factor, and the resulting arithmetic intensity.
(d) Where does that put you on your GPU's roofline?
(e) Now 32x32 tiles. Recompute. Shared memory needed? Occupancy?
(f) Extrapolate: what tile size would be needed to be fully compute-bound in
    fp32 on your GPU? Does it fit in shared memory? If not, what else must
    change -- and that question is the opening of Chapter 11.

---

## Exercises to run

- [`06-transpose`](../exercises/06-transpose) - three versions, measured
  against the copy ceiling. The bar is 85%.

---

## What you should now believe

- Shared memory is a scratchpad you control, ~20x faster than global.
- It does two jobs: reshaping access patterns, and enabling reuse.
- Tiling = cooperative load, barrier, compute, barrier.
- 32 banks of 4 bytes; column access conflicts 32-way; padding or swizzling
  fixes it.
- Reuse factor divides Q and multiplies arithmetic intensity, which is how you
  move along the roofline.
- Barriers cost real time, and avoiding the stall is what double buffering is
  for.

Next: [Chapter 07 - Reductions](07-reductions.md).
