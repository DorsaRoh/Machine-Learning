# Chapter 12 - GEMM II: closing on cuBLAS

> You are at 50-70% of cuBLAS. The remaining gap is made of five or six
> specific, nameable things. This chapter names them.

---

## 1. Where the missing 30% is

Profile your v5 and you will find the time going to, roughly in order:

1. **Barrier stalls** -- every K-iteration, all warps wait for the tile load.
2. **Shared-memory bank conflicts** in the inner loop.
3. **Not enough bytes in flight** during the load phase.
4. **Instruction overhead** -- address arithmetic competing with FMA issue.
5. **Tail and quantization effects** -- partial tiles, partial waves.
6. **Suboptimal tile size** for this particular shape.

Each has a known fix. Work them in order of measured cost, not in the order
they appear here.

---

## 2. Double buffering

The structural problem with the Chapter 11 loop:

```
load tile k    -> barrier -> compute tile k -> barrier -> load tile k+1 -> ...
```

During the load, the FMA pipes are idle. During the compute, the memory system
is idle. You alternate between being memory-bound and compute-bound, achieving
neither roof.

Double buffering overlaps them:

```
load tile 0 into buffer 0
for k:
    issue load of tile k+1 into buffer (k+1)%2     <- does not block
    compute on buffer k%2                           <- overlaps with the load
    barrier
```

Now the load for the next iteration is in flight while you do the math for this
one. Cost: twice the shared memory. Benefit: typically 10-20%, and more on
memory-latency-bound shapes.

The subtlety is that a plain `tile[buf][i] = A[...]` load goes global -> register
-> shared, so the *register* arrival still blocks. True overlap wants the data
to move without passing through your registers, which is `cp.async` (Chapter
14) on Ampere and later. On older hardware the usual trick is to stage in
registers explicitly:

```
registers <- global for tile k+1   (issued early, consumed late)
compute on shared tile k
shared <- registers for tile k+1
barrier
```

Writing that out and seeing why it overlaps is exercise 12.2.

---

## 3. Vectorized everything

**Global loads** as `float4`: a quarter of the load instructions, 4x the bytes
in flight per instruction. Requires 16-byte alignment and tile dimensions
divisible by 4.

**Shared loads** as `float4` in the inner loop: `FLOAT4(As[k][ty*TM])` fetches
four A values in one `LDS.128`. This matters more than people expect -- the
inner loop issues an enormous number of shared loads, and quartering them frees
issue slots for FMAs.

**The conflict interaction.** `float4` shared accesses are serviced in a
different pattern than scalar ones (each is 4 banks wide), so the `+1` padding
trick that fixes scalar conflicts *breaks* `float4` alignment. This is exactly
why real kernels use XOR swizzles rather than padding. Chapter 06 exercise 6.3
asked you to derive one; here is where you need it.

---

## 4. Transposing the A tile

The inner loop wants, for a fixed `k`, the `TM` values `A[i..i+TM][k]` -- a
*column* of the A tile. Stored row-major as `As[BK][BM]`... wait, stored how?

- Store `As` as `[BM][BK]` (natural): the column read strides by BK. Bank
  conflicts, and no `float4`.
- Store `As` as `[BK][BM]` (transposed on the way in): the column read becomes
  contiguous. Conflict-free, `float4`-able.

So you transpose A *while loading it into shared memory*, which costs nothing
because you are rearranging data you are moving anyway. Chapter 06's transpose,
embedded as a free side effect of a load.

This is the kind of move that distinguishes a fast GEMM: you are not adding
work, you are choosing the layout at the one moment the data is in flight and
the choice is free.

---

## 5. Warp tiling

Between the block tile and the thread tile there is a level you skipped: the
**warp tile**.

The 32 threads of a warp should compute a *contiguous rectangle* of C, not a
scattered set. Why: when a warp reads its A fragment and B fragment from shared
memory, a compact rectangular footprint means those reads hit few banks and can
broadcast. A scattered assignment means more shared transactions for the same
data.

A typical arrangement: a 128x128 block tile, split into 2x4 warps each owning a
64x32 warp tile, each warp's 32 threads owning 8x8 thread tiles within it.
Three nested levels of 2D decomposition, and CUTLASS's `efficient_gemm.md`
describes exactly this.

Getting the warp level right is usually worth 10-15% over the naive "thread t
owns row t/16, column t%16" assignment.

---

## 6. Tails, waves and shape

**Partial tiles.** M = 4000 with BM = 128 gives 31.25 tiles. Options: guard
loads with bounds checks and write zeros into the shared tile (clean, costs a
predicate per load), or run a separate tail kernel. Guarding is standard; make
sure the *inner loop* has no guards, only the load.

**Wave quantization.** `grid = ceil(M/BM) * ceil(N/BN)` blocks against
`SMs * blocks_per_SM` slots. If your grid is 1.1 waves, you run at 55%
efficiency. For a 108-SM GPU with 2 blocks per SM, grids near multiples of 216
are good and 217 is terrible. **Split-K** -- splitting the K loop across blocks
and reducing partials afterwards -- exists mainly to fill the machine when M
and N are small, and it is what cuBLAS switches to for skinny shapes.

**Shape-dependent kernels.** A 4096^3 GEMM and a `[1, 4096] x [4096, 4096]`
GEMV want completely different kernels. The second is memory-bound (Chapter 01,
LLM decode) and a tiled GEMM kernel is the wrong tool entirely. Libraries
dispatch on shape; so should you.

---

## 7. Autotuning

The parameter space:

```
BM, BN in {32, 64, 128, 256}
BK     in {8, 16, 32, 64}
TM, TN in {2, 4, 8, 16}
double buffer: yes/no
vector width: 1, 2, 4
```

Hundreds of valid combinations, and the best one depends on the GPU, the dtype
and the shape. There is no formula. What you do:

1. Prune with the constraints from Chapter 11 section 5 (shared memory,
   registers, divisibility). Most combinations die here.
2. Benchmark the survivors on the shapes you care about.
3. Cache the winner per (GPU, shape bucket).

This is what CUTLASS profiler does, what Triton's `@autotune` does, and what
cuBLAS does at build time with heuristics at runtime. Build a small version in
exercise 12.5 -- it is a genuinely useful tool to own.

---

## 8. What cuBLAS does that you will not

Honest accounting of the last few percent:

- **Hand-tuned SASS scheduling.** cuBLAS kernels are tuned at the assembly
  level, interleaving loads and FMAs to the exact dual-issue pattern of each
  architecture. `ptxas` is good; a human with a profiler and a month is better.
- **Hundreds of specialised kernels** with runtime heuristic dispatch.
- **Architecture-specific instructions** that appear in SASS but not in CUDA C,
  including specific `LDGSTS` patterns and cache-policy hints.
- **Split-K and stream-K** variants for awkward shapes.

Reaching 70-80% of cuBLAS with a well-structured CUDA C kernel is a genuinely
good result and is the bar for this course. 90%+ requires CUTLASS, and CUTLASS
exists precisely so you do not have to do the above by hand.

---

## 9. Common failure modes

**Double buffering made it slower.** Shared memory doubled, occupancy halved,
and you did not gain overlap because the load still blocks on registers. Check
blocks-per-SM before and after.

**`float4` shared loads broke correctness.** Alignment: the shared array must
be 16-byte aligned and your index must be a multiple of 4.

**Bank conflicts reappeared after adding `float4`.** Padding and vectorized
access are incompatible. Swizzle.

**Great at 4096, poor at 1024.** Wave quantization. Count your blocks against
your SM count.

**Great at 4096^3, terrible at `[4096, 4096] x [4096, 16]`.** Skinny N. You
need split-K or a different kernel. This is not a bug.

**Registers jumped and performance collapsed after a small change.** The
compiler stopped unrolling, and your accumulator array went to local memory.
Check for `LDL`/`STL` in the SASS.

---

## 10. Exercises

### 12.1 - Where is your 30%?

Profile your v5 from `11-gemm-ladder` and produce a budget:

```
total time                         100%
  stall barrier                    ___%
  stall short scoreboard (shared)  ___%
  stall long scoreboard (global)   ___%
  stall math throttle              ___%
  stall not selected               ___%
bank conflicts (shared ld/st)      ___ per inner-loop iteration
achieved occupancy                 ___%
registers per thread               ___
spills                             ___
```

Rank the fixes by expected value *before* implementing any of them. Then
implement in that order and record whether your ranking was right.

### 12.2 - Design double buffering

(a) Draw the timeline of load/compute for the single-buffered loop. Mark idle
    periods for the memory system and the FMA pipes.
(b) Draw the double-buffered version.
(c) Why does a naive `shared[buf][i] = A[...]` not fully overlap? Where does
    the data physically go on its way?
(d) Write the register-staged variant that does overlap on pre-Ampere hardware.
(e) Shared memory cost. New occupancy. Is it still enough?
(f) Predict the speedup. Implement. Compare.

### 12.3 - The swizzle, for real

Your inner loop reads `float4` from the A tile and the B tile.

(a) With `As[BK][BM]` and no padding, which banks does warp 0 touch in a
    `float4` read? Conflicts?
(b) Why does `+1` padding break `float4` here?
(c) Design an XOR swizzle that keeps 16-byte alignment and avoids conflicts.
(d) Count the extra instructions it costs per access.
(e) Implement and measure the conflict counter before and after.

### 12.4 - Wave quantization

On your GPU, with 128x128 tiles:

(a) Blocks for M = N = 4096. Blocks per SM at your occupancy. Waves. Efficiency
    of the last wave.
(b) Same for M = N = 4000.
(c) Same for M = N = 1024.
(d) Sweep N from 1024 to 8192 in steps of 128, measure TFLOP/s, and plot. The
    sawtooth you get is wave quantization. Mark the peaks and explain their
    spacing.
(e) At which shapes would split-K help? Implement it for one and measure.

### 12.5 - Build an autotuner

(a) Template your kernel on (BM, BN, BK, TM, TN).
(b) Generate the valid combinations at compile time, pruning by shared memory,
    register estimate and divisibility.
(c) Benchmark all survivors on 4096^3 and report the top 5.
(d) Repeat for `[1024, 1024] x [1024, 1024]` and for `[8192, 512] x [512,
    8192]`. Is the winner the same? How much do you lose by using one fixed
    configuration everywhere?
(e) This number -- the cost of not autotuning -- is the argument for Triton's
    `@autotune` and for CUTLASS's profiler. Write it down.

### 12.6 - Read a real kernel

Open CUTLASS's `efficient_gemm.md` and one of its SM80 GEMM kernels.

(a) Identify the threadblock tile, warp tile, and instruction tile.
(b) Find where it stages global -> shared. Which instruction?
(c) Find its swizzle. Which function?
(d) Find its pipeline depth. How many stages, and why that number?
(e) Name one thing it does that your kernel does not, and estimate its worth.

Do this only after your own v5 passes. Beforehand it is noise; afterwards it is
a conversation with someone who solved your problem.

---

## Exercises to run

- `11-gemm-ladder` v5, now aiming past 70% toward 85% of cuBLAS.

---

## What you should now believe

- The last 30% is a specific, enumerable list, not mystery.
- Double buffering overlaps load and compute, and it needs the load to bypass
  registers to do it fully.
- Vectorized shared access and swizzled layouts go together; padding and
  `float4` do not.
- The warp tile is a real level of the hierarchy, not an implementation detail.
- Wave quantization and shape dispatch explain most "why is it slow at this
  size" questions.
- 70-85% of cuBLAS is an excellent hand-written result.

Next: [Chapter 13 - Tensor Cores](13-tensor-cores.md), where the FMA stops
being the unit of work.
