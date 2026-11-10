# Glossary

Defined once, properly. Terms are grouped by the layer they belong to, because
the single most common source of confusion in GPU programming is mixing up the
*programming model* (what you write) with the *execution model* (what the chip
does) with the *hardware* (what exists in silicon).

---

## The programming model: what you write

**Kernel** - a function marked `__global__`, launched from the host, executed
by many threads at once. One source body, N instances.

**Thread** - one instance of the kernel body. Has its own registers and its own
program counter (on Volta and later). Cheap: a GPU keeps tens of thousands
resident.

**Block** (CTA, Cooperative Thread Array) - a group of threads that are
guaranteed to run on the *same* SM, can share a scratchpad (`__shared__`), and
can synchronise with `__syncthreads()`. Up to 1024 threads. This is the only
unit of the model in which cooperation is cheap.

**Grid** - all blocks of one launch. Blocks in a grid cannot synchronise with
each other (except via cooperative launch, Chapter 18) and may run in any
order, including entirely sequentially. Any algorithm requiring inter-block
ordering is either wrong or needs a different structure.

**Cluster** (sm_90+) - a group of blocks guaranteed co-resident on the same GPC,
able to access each other's shared memory. Hopper's answer to "blocks cannot
cooperate".

**`__device__` / `__global__` / `__host__`** - compiled for GPU and callable
from GPU / compiled for GPU and callable from CPU / ordinary CPU function.

**Launch configuration** - `kernel<<<grid, block, smem_bytes, stream>>>(...)`.
Four decisions, all of which have performance consequences.

---

## The execution model: what the chip does with it

**Warp** - 32 threads that issue instructions together. The real unit of
execution; the thread is a fiction maintained for your convenience. Every
performance question eventually becomes a question about what one warp does.

**SIMT** (Single Instruction, Multiple Thread) - the warp executes one
instruction at a time across 32 lanes, with a mask for inactive lanes. Like
SIMD, except each lane has its own address and the divergence handling is
automatic (and expensive).

**Divergence** - threads within a warp taking different paths at a branch. The
warp serialises: executes the `if` side with some lanes masked off, then the
`else` side with the others masked off. Cost is additive, not maximum.

**Lane** - a thread's index within its warp, `threadIdx.x % 32`.

**Independent thread scheduling** (Volta+) - each thread has its own PC, so
threads in a warp can be at different instructions. Makes some lock patterns
legal that were deadlocks before, and makes warp-synchronous programming
without explicit `__syncwarp()` *illegal*.

**Occupancy** - resident warps per SM divided by the hardware maximum. A
measure of how much latency the SM can hide, not a measure of speed. High
occupancy with no work to do is worthless; low occupancy with enough
instruction-level parallelism is fine.

**Latency hiding** - when a warp stalls on a memory load, the scheduler issues
from a different resident warp. This, not caching, is the GPU's primary
strategy for tolerating memory latency.

**Tail effect** - the last partial wave of blocks leaves most SMs idle. A grid
of 65 blocks on a 64-SM GPU takes roughly twice as long as 64 blocks.

**Wave** - one full round of concurrently resident blocks across all SMs.

---

## Memory

**Global memory** - the big off-chip pool (HBM or GDDR). High bandwidth,
~400-800 cycles of latency. Everything starts and ends here.

**L2 cache** - a few MB, shared by all SMs, backs global memory. The reason
repeated small-buffer benchmarks lie to you.

**L1 / shared memory** - on-chip per SM, usually the same physical SRAM split
between a hardware-managed cache (L1) and a software-managed scratchpad
(`__shared__`). ~20-30 cycle latency.

**Registers** - per-thread, fastest storage, allocated by the compiler. A
finite pool per SM; using more per thread means fewer resident warps.

**Register spilling** - the compiler ran out of registers and put variables in
"local memory", which despite the name lives in *global* memory. Disastrous if
it happens in an inner loop. Check with `ptxas -v`.

**Local memory** - per-thread memory that is physically global memory. Used for
spills and for arrays indexed dynamically. Not local to anything.

**Constant memory** - 64 KB, read-only, broadcast-optimised. Fast when all
threads in a warp read the *same* address, slow when they do not.

**Texture / `__ldg` / read-only cache** - a path through the texture unit for
read-only data with its own cache and spatial locality handling. Mostly
subsumed by `const __restrict__` on modern hardware.

**Coalescing** - the memory system servicing a warp's 32 addresses with the
minimum number of transactions. Perfect: 32 consecutive 4-byte addresses in one
128-byte line = 4 sectors. Worst case: 32 scattered addresses = 32 sectors,
i.e. 8x the traffic for the same data.

**Sector** - the 32-byte granule the memory system actually moves. The unit in
which to count waste. A cache line is 128 bytes = 4 sectors.

**Bank conflict** - shared memory is 32 banks of 4-byte words. Two threads in a
warp hitting different addresses in the *same* bank serialise. Same address in
the same bank is a broadcast and is free.

**Swizzle** - permuting a shared-memory layout (usually with XOR) so that the
access pattern stops landing in the same bank. The general version of the `+1`
padding trick.

**`cp.async`** (sm_80+) - copy global to shared without going through
registers, asynchronously, so the SM can compute while the copy is in flight.

**TMA** (Tensor Memory Accelerator, sm_90+) - a hardware copy engine that moves
whole multidimensional tiles global <-> shared, driven by a descriptor, freeing
warps from address arithmetic entirely.

---

## Math and throughput

**FLOP** - one floating-point operation. A matrix multiply of (M,K)x(K,N) is
`2*M*N*K` FLOPs (a multiply and an add per inner-product term).

**Arithmetic intensity** - FLOPs performed per byte moved from global memory.
The single most important number about a kernel. Determines which side of the
roofline you live on.

**Roofline** - achievable performance = min(peak FLOP/s, bandwidth x arithmetic
intensity). The knee is where the two lines cross; kernels left of it are
memory-bound, right of it compute-bound.

**Memory-bound** - runtime determined by bytes moved. Almost all ML kernels
except GEMM and convolution: elementwise, normalization, softmax, attention at
short sequence lengths, LLM decode.

**Compute-bound** - runtime determined by math throughput. GEMM at large sizes,
convolution, attention at long sequence lengths.

**Latency-bound** - neither pipe is saturated; there simply is not enough work
in flight. Symptom: low achieved bandwidth *and* low compute utilisation. Fixed
with more parallelism, more ILP, or fewer dependent instructions.

**Tensor Core** - a hardware unit performing a small matrix multiply-accumulate
(e.g. 16x8x16) in one instruction. 8-16x the FLOP/s of the regular FP32 pipes,
and the reason modern GEMM has nothing to do with `a*b+c` loops.

**TF32 / BF16 / FP16 / FP8** - reduced-precision formats. TF32 has fp32 range
with 10 mantissa bits; BF16 has fp32 range with 7; FP16 has more mantissa but
narrow range and overflows at 65504; FP8 comes in E4M3 and E5M2 flavours. Range
matters more than precision for training stability.

**MAC / FMA** - multiply-accumulate, `d = a*b + c` as one instruction with one
rounding. Two FLOPs.

---

## Synchronisation

**`__syncthreads()`** - barrier for all threads in a block. Must be reached by
every thread in the block; a `__syncthreads()` inside a divergent branch is
undefined behaviour and in practice hangs or corrupts.

**`__syncwarp(mask)`** - barrier for the named lanes of a warp. Required on
Volta+ wherever you previously relied on implicit warp-synchronous execution.

**Warp shuffle** (`__shfl_sync`, `__shfl_down_sync`, `__shfl_xor_sync`) -
read another lane's register directly. No shared memory, no barrier. The
backbone of fast reductions and scans.

**Atomic** - read-modify-write that cannot be interleaved. Fast when
contention is low (they execute at L2, or in shared memory at the SM), lethal
when every thread hits the same address.

**Memory fence** (`__threadfence`, `__threadfence_block`) - orders your memory
operations as seen by others. Not a barrier: it does not make anyone wait, it
only establishes ordering.

**Cooperative groups** - a typed API for the above, including grid-wide
synchronisation when launched cooperatively.

---

## Tooling

**PTX** - NVIDIA's virtual ISA. Portable, JIT-compiled to SASS by the driver.
What inline `asm` in CUDA is written in.

**SASS** - the actual machine code for a specific architecture. What runs. Read
it with `cuobjdump -sass` when the profiler stops answering your question.

**`ptxas`** - the backend compiler PTX -> SASS. `-Xptxas -v` reports registers,
shared memory and spills per kernel.

**Nsight Compute (`ncu`)** - per-kernel profiler: counters, speed-of-light,
stall reasons, source-level attribution.

**Nsight Systems (`nsys`)** - timeline profiler: kernels, copies, CPU activity,
gaps. The tool for "my GPU is idle 40% of the time".

**compute-sanitizer** - memcheck (out of bounds), racecheck (shared memory
races), initcheck (uninitialised reads), synccheck (barrier misuse).

**cuBLAS / cuDNN / CUTLASS** - NVIDIA's GEMM/conv libraries, and the open C++
template library that shows you how the fast ones are built. CUTLASS is the
best reading material in this field.

**Triton** - a Python DSL where you program at *block* granularity and the
compiler handles the intra-block mapping, vectorisation and shared memory.
