# Chapter 03 - Memory hierarchy and coalescing

> If you only ever learn one chapter of this course properly, make it this one.
> Coalescing is worth up to 8x, costs nothing to get right, and is invisible in
> code that looks perfectly reasonable.

---

## 1. The hierarchy, with numbers

```
            capacity        latency      bandwidth      scope
registers   256 KB/SM       ~1 cyc       enormous       one thread
shared/L1   up to 228 KB/SM ~25 cyc      ~20 TB/s       one block
L2          4-72 MB         ~200 cyc     ~5 TB/s        whole GPU
HBM/global  16-80 GB        ~500 cyc     1-3 TB/s       whole GPU + host
host        TBs             ~10 us       ~25 GB/s       CPU
```

Each step down is roughly 10x slower and 1000x larger. The entire craft of
kernel optimisation is: **get data to the smallest level that will hold it, and
use it as many times as possible while it is there.**

Note the ratio between shared memory and global: ~20x in bandwidth and ~20x in
latency. That factor is the prize that tiling (Chapter 06) wins, and it is why
GEMM is written the way it is.

---

## 2. What a memory instruction actually does

A warp executes `float v = a[idx];`. All 32 lanes issue their load **as one
instruction**. The memory system now has 32 addresses to service.

It does not fetch 32 individual floats. It works in **32-byte sectors**. It
gathers the 32 addresses, finds the distinct sectors they fall into, and
issues one transaction per sector.

```
addresses (bytes)   : 0, 4, 8, ..., 124        (32 consecutive floats)
distinct sectors    : [0,32) [32,64) [64,96) [96,128)   = 4 sectors
bytes moved         : 128.  bytes wanted: 128.  efficiency: 100%
```

versus

```
addresses (bytes)   : 0, 128, 256, ..., 3968   (stride 32 floats)
distinct sectors    : 32 different ones
bytes moved         : 1024.  bytes wanted: 128.  efficiency: 12.5%
```

Same instruction. Same number of useful floats. **Eight times the traffic**,
and therefore eight times the runtime for a memory-bound kernel.

This is coalescing. The rule:

> A warp's 32 addresses should fall into as few 32-byte sectors as possible.
> For 4-byte loads, the ideal is 4 sectors; the worst case is 32.

The profiler reports this directly as `sectors per request`. 4 is perfect, 32
is catastrophic, and the ratio between what you have and 4 is your available
speedup.

### The patterns

```
PERFECT: lane i reads a[base + i]
  |====|====|====|====|      4 sectors, 128 bytes, all used

OFFSET: lane i reads a[base + i + 1]  (base not 128-aligned)
  |..==|====|====|====|==..|  5 sectors. ~20% waste. Usually tolerable.

STRIDE 2: lane i reads a[base + 2*i]
  |=.=.|=.=.|...               8 sectors, half of every one wasted. 2x slow.

STRIDE 32: lane i reads a[base + 32*i]
  |=...|  |=...|  |=...|       32 sectors, 1/8 of each used. 8x slow.

RANDOM: lane i reads a[perm[i]]
  scattered                    up to 32 sectors. 8x slow, plus TLB pressure.

BROADCAST: every lane reads a[base]
  |=...|                       1 sector. Free. The hardware broadcasts.
```

Two things worth internalising:

- The penalty **saturates at 32x stride**. Going from stride 32 to stride 1000
  costs nothing extra (one sector per lane either way), though TLB misses start
  to appear.
- **Broadcast is free.** All 32 lanes reading the same address is one sector.
  This is why reading a scalar parameter inside a loop costs nothing.

---

## 3. Where bad patterns come from

Nobody writes `a[32*i]` on purpose. Coalescing dies in these ways:

### Array of Structures

```cuda
struct Particle { float x, y, z, vx, vy, vz; };   // 24 bytes
Particle* p;
float xi = p[i].x;     // lane i reads byte 24*i -> stride 24, ~6x waste
```

versus Structure of Arrays:

```cuda
struct Particles { float *x, *y, *z, *vx, *vy, *vz; };
float xi = p.x[i];     // lane i reads byte 4*i -> perfect
```

**SoA is the default layout on a GPU.** This is also why ML frameworks store
tensors as dense typed planes and not as arrays of structs.

### Wrong index as fastest-varying

```cuda
// column-major traversal of a row-major matrix
int row = blockIdx.x * blockDim.x + threadIdx.x;
for (int col = 0; col < N; ++col)
    sum += A[row * N + col];     // lane i reads A[i*N + col]: stride N
```

Consecutive lanes are N elements apart. For N = 4096 that is 16 KB apart: one
sector each, 8x waste. Swapping which index the thread owns fixes it entirely.

### Thread-chunked loops

```cuda
// BAD: each thread takes a contiguous chunk
int start = tid * chunk;
for (int j = 0; j < chunk; ++j) sum += a[start + j];
```

At any instant, lane 0 is at `0`, lane 1 is at `chunk`, lane 2 at `2*chunk`.
This is maximally strided. The grid-stride loop from Chapter 02 does the same
total work with perfect coalescing because at every step the warp's lanes are
adjacent. **A loop's coalescing is determined by what the warp does in one
iteration, not by what one thread does over all iterations.**

### 2D blocks indexed the wrong way round

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
out[x * height + y] = in[x * height + y];   // x is fastest-varying in the warp
                                            // but strides by `height` in memory
```

Swap to `[y * width + x]` and the same kernel runs 8x faster.

---

## 4. Vectorized access

A single thread can load 8 or 16 bytes in one instruction:

```cuda
float4 v = reinterpret_cast<const float4*>(a)[i];
// v.x v.y v.z v.w, one LDG.E.128 instruction
```

Why this helps even though the total bytes are unchanged:

- **Fewer instructions** for the same data: a quarter of the address
  arithmetic, a quarter of the load-issue slots.
- **More bytes in flight per thread.** Little's Law again: one `float4` load
  outstanding is 16 bytes of concurrency instead of 4. This is often worth
  10-20% on a bandwidth-bound kernel that was short of concurrency.
- A warp doing `float4` loads covers 512 bytes contiguously = 16 sectors, all
  fully used.

The requirements, both of which bite:

1. **Alignment.** `float4` requires a 16-byte-aligned address. `cudaMalloc`
   returns 256-byte-aligned pointers, so the base is fine, but `a + 1` is not.
   Misalignment is a hard runtime error, `misaligned address`.
2. **Divisibility.** Your element count must be a multiple of 4, or you need a
   scalar tail loop.

Check it worked: `cuobjdump -sass` should show `LDG.E.128`. If you see
`LDG.E.32`, the compiler did not believe your alignment.

---

## 5. Read-only hints

```cuda
__global__ void k(const float* __restrict__ a, float* __restrict__ out)
```

`const` + `__restrict__` together tell the compiler two things: this data does
not change during the kernel, and no other pointer aliases it. With that
guarantee it can route loads through the read-only cache (`LDG.CONSTANT` /
`__ldg`), cache values in registers across unrelated stores, and reorder loads
freely.

Free performance. Put `const __restrict__` on every input pointer and
`__restrict__` on every output pointer, every time, as a reflex.

---

## 6. Writes, and the cost you did not count

Writing to memory you have not read usually triggers a **read-modify-write**:
the hardware fetches the cache line, modifies part of it, and writes it back.
A kernel that only writes N bytes can move 2N.

Two mitigations:

- **Write full cache lines.** If a warp writes 32 consecutive floats, the whole
  128-byte line is covered and no fetch is needed. Coalesced writes avoid the
  problem automatically. Another reason coalescing matters.
- **Streaming / non-temporal stores** (`__stcs`, `st.global.cs`) tell the cache
  this data will not be re-read, so do not pollute L2 with it. Useful for large
  outputs that nothing downstream touches soon.

When your measured `dram__bytes_write` is about twice your hand count, this is
why.

---

## 7. L2 and the benchmarking lie

L2 is 4-72 MB, shared by every SM. It matters in two ways.

**It makes benchmarks lie.** Run a kernel 100 times over a 4 MB buffer on an
A100 (40 MB of L2) and iterations 2-100 never touch HBM. You will measure
"5 TB/s" and believe it. This is why `bench.cuh` flushes L2 between timed
iterations and why your working sets in this course are deliberately large.

**It can be steered.** On Ampere and later you can reserve a fraction of L2 for
a specific address range:

```cuda
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

Real use: pinning a small frequently-read table (embedding lookup, attention
sink, quantization scales) while streaming data flows past it. Niche, but when
it applies it is a large win.

---

## 8. The canonical problem: transpose

`B[j][i] = A[i][j]`. No arithmetic at all; pure memory. Perfect test case.

```cuda
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];
}
```

Trace one warp. Lanes vary in `x`, `y` fixed.

- **Read** `in[y*n + x]`: addresses are consecutive in x. **Coalesced.** 4
  sectors.
- **Write** `out[x*n + y]`: consecutive lanes are `n` elements apart.
  **Strided.** 32 sectors.

So a perfect copy moves `2 * 4 = 8` sectors per warp-pair of instructions and
this moves `4 + 32 = 36`. Expect roughly 4.5x less than copy bandwidth, which
is exactly what you will measure.

And here is the thing that makes the next chapter necessary: **there is no way
to index your way out of this**. Reads and writes want opposite layouts. Fix
one, break the other. The only escape is a staging area where the block can
reorder data internally -- shared memory, Chapter 06. Transpose is the smallest
problem that forces the idea, which is why every course uses it.

---

## 9. Common failure modes

**Achieving 1/8 of copy bandwidth.** Stride-32 access. Find which index varies
with `threadIdx.x`.

**Achieving 1/2.** Stride 2, or an AoS with two fields, or write-allocate
you did not count.

**Fast for small n, collapses past a size.** You were in L2. Compare your
working set to the L2 size.

**`float4` version no faster.** Check the SASS for `LDG.E.128`. Check your
pointer is really 16-byte aligned. Check you were not already saturating
bandwidth, in which case there is nothing to win.

**`misaligned address`.** A vectorized load from an offset base. `a + 1` as
`float4*` is illegal.

**2D kernel 8x slow.** `threadIdx.x` mapped to rows instead of columns.

**Transpose seems fine at n=256.** The whole matrix is 256 KB and lives in L2.
Measure at n >= 8192.

---

## 10. Exercises

### 3.1 - Count the sectors

For each, with `float` data and 32 lanes, state the number of 32-byte sectors
and the efficiency:

(a) `a[i]`
(b) `a[i + 1]` with `a` 256-byte aligned
(c) `a[2*i]`
(d) `a[i/2]`
(e) `a[31 - i]`
(f) `a[(i * 7) % 32]`
(g) `a[0]`
(h) `float4` at `a4[i]`
(i) `a[i * 1024]`
(j) `a[i]` where `a` is `half` (2 bytes)
(k) `a[i].x` for `struct {float x,y;} a[]`

For (e) and (f): does the hardware care about order, or only about the set of
addresses? Does that surprise you? What does it imply for a kernel that
shuffles data within a warp's worth of contiguous memory?

### 3.2 - Predict the stride curve

You will measure `y[i] = x[i * stride]` for stride 1..64.

(a) Sketch the curve of achieved bandwidth (useful bytes per second) vs stride.
(b) Where does it plateau, and at what fraction of peak?
(c) Now sketch *total* bytes moved by the hardware vs stride. Different shape?
(d) At stride 1024, does performance drop further? Why or why not? What new
    effect appears at very large strides?

Then run `03-stride-study` and compare. Explaining any discrepancy is the
exercise.

### 3.3 - Rewrite for coalescing

Each snippet is slow. State the sectors per request, then rewrite the access
pattern (not the algorithm) to fix it. Do not run anything.

```cuda
// (a)
struct RGB { unsigned char r, g, b; };
__global__ void grey(const RGB* in, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = (in[i].r + in[i].g + in[i].b) / 3;
}

// (b) row sums
__global__ void rowsum(const float* A, float* s, int M, int N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    float t = 0;
    for (int c = 0; c < N; ++c) t += A[r * N + c];
    s[r] = t;
}

// (c)
__global__ void shift(const float* in, float* out, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[(i + k) % n];
}

// (d) batched: [batch][feature], one thread per batch element
__global__ void norm(const float* x, float* y, int B, int F) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0;
    for (int f = 0; f < F; ++f) s += x[b * F + f] * x[b * F + f];
    s = rsqrtf(s);
    for (int f = 0; f < F; ++f) y[b * F + f] = x[b * F + f] * s;
}
```

For (b) and (d), the fix is not an index change -- it is a change of *who does
what*. That reassignment is the single most common restructuring in ML kernels,
and you will do it again in Chapter 10.

### 3.4 - The transpose budget

For an 8192 x 8192 fp32 transpose on your GPU:

(a) Compulsory bytes.
(b) Best possible time at your measured copy ceiling.
(c) Sectors moved per warp by the naive version, read and write separately.
(d) Predicted naive time.
(e) Predicted naive bandwidth as the harness computes it (useful bytes / time).
(f) Which of read or write is the problem, and what would it take to fix that
    one without breaking the other?

Then run `03-transpose-naive`. You are not asked to make it fast here. You are
asked to *account* for the loss to within 20%.

---

## Exercises to run

- [`03-stride-study`](../exercises/03-stride-study) - measure the coalescing
  penalty curve and explain every feature of it.
- [`03-transpose-naive`](../exercises/03-transpose-naive) - measure the loss,
  attribute it, and set up Chapter 06.

---

## What you should now believe

- The memory system moves 32-byte sectors, and a warp's addresses determine how
  many.
- Coalescing is worth up to 8x and is free.
- SoA, not AoS. `threadIdx.x` maps to the fastest-varying memory dimension.
- `const __restrict__` always; `float4` when aligned.
- L2 will lie to you in benchmarks unless you flush it.
- Transpose cannot be fixed by indexing, which is why shared memory exists.

Next: [Chapter 04 - Occupancy, warps and registers](04-occupancy.md).
