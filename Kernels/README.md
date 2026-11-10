# GPU Kernels: a masterclass

From "what is a thread" to writing a fused attention kernel and reading the
machine code it compiles to.

This is a course, not a reference. It is built around one belief: you do not
learn to write kernels by reading kernels. You learn by predicting a number,
measuring it, being wrong, and finding out why. Every chapter here exists to
give you a **cost model** -- a way to look at a loop and say "that will run at
340 GB/s" before you compile it. Every exercise exists to find out whether your
cost model is real.

---

## The rules

**1. You will not be given answers.**
No chapter contains a finished kernel for anything you are asked to write.
Exercises ship as skeletons with `// TODO`. Hints are questions, never code.
There are no solution files in this repository and there never will be. If you
go and find one on the internet, you have bought a lap time instead of learning
to drive, and Chapter 19 will be impossible.

**2. Predict before you measure.**
Every exercise brief starts with a prediction box. Write down the number --
GB/s, TFLOP/s, or a ratio -- *before* you run anything. The gap between your
prediction and reality is the entire curriculum. An engineer who gets the right
answer without knowing why they were about to get the wrong one has learned
nothing transferable.

**3. Two gates, both mandatory.**
Correctness is checked against a float64 reference. Performance is checked
against a ceiling *measured on your GPU*, so the bar is equally fair on a T4
and an H100. "It works" is half the job. A correct kernel at 12% of bandwidth
is a bug with better manners.

**4. Profile before you optimise.**
From Chapter 16 onward, a performance change that you cannot explain in terms
of a counter -- sectors per request, stall reason, achieved occupancy -- does
not count, even if the number went up. Guess-and-check is not engineering, and
it stops working the moment the kernel gets complicated.

---

## How to use this

The loop for every exercise:

```
read the chapter -> write your prediction -> write the kernel -> make run
        ^                                                            |
        |                                                            v
   understand why  <----  profile / sanitize  <----  gate says NOT YET
```

Do not move to the next chapter with a failed gate behind you. The course is
strictly cumulative: tiling in Chapter 06 is the same idea as the GEMM in
Chapter 11 is the same idea as FlashAttention in Chapter 19. A soft spot early
becomes a wall later.

Every exercise through Chapter 22 is a C++/CUDA directory driven by `make run`.
The three Part VI exercises are Python: `23-triton-ladder` and
`24-torch-extension` run with `python bench.py`, and `26-capstone` has no
harness at all because you write it.

**Setup:** [setup/GETTING_STARTED.md](setup/GETTING_STARTED.md). You need a
rented NVIDIA GPU; a free Colab T4 covers most of the course.

**No background in C, pointers or performance?** Then start one part earlier,
in [Part -1](prerequisites/README.md), which assumes nothing and runs entirely
on your own laptop with no GPU at all.

**Pace:** roughly one chapter per sitting for Parts 0-II, two sittings per
chapter for Parts III and V. The whole thing is a few months of evenings. There
is no faster honest version.

---

## Syllabus

### Part -1 - Prerequisites (optional, on your laptop)

Start here if "pointer", "row-major" or "arithmetic intensity" are not already
familiar. Nine chapters and six exercises that assume you know nothing, and
that run on your own machine with a plain C++ compiler — no GPU, no CUDA, no
rented instance.

| # | Chapter | What you walk away with |
|---|---|---|
| P0 | [What a computer actually is](prerequisites/p0-what-a-computer-is.md) | Bits, bytes, addresses, memory as one enormous array, why cache exists |
| P1 | [C essentials](prerequisites/p1-c-essentials.md) | Compiling and running, types, functions, loops, `printf` |
| P2 | [Pointers and memory](prerequisites/p2-pointers-and-memory.md) | Addresses, dereferencing, stack vs heap, `const`, the classic mistakes |
| P3 | [Arrays, indexing and strides](prerequisites/p3-arrays-and-indexing.md) | Flattening 2D and 3D, row-major, strides, tiles. **The important one** |
| P4 | [How numbers actually work](prerequisites/p4-numbers.md) | Binary, overflow, floating point, why `a+b+c != c+b+a` |
| P5 | [Just enough C++](prerequisites/p5-cpp-for-cuda.md) | Only the parts CUDA uses: headers, structs, `std::vector`, templates, lambdas |
| P6 | [The toolchain](prerequisites/p6-toolchain.md) | Compiler, linker, `make`, sanitizers, reading an error without panic |
| P7 | [The math you need](prerequisites/p7-the-math.md) | Dot products, matmul, reductions, softmax, normalisation |
| P8 | [Performance arithmetic](prerequisites/p8-performance-arithmetic.md) | Bytes, FLOPs, GB/s, estimating a loop's cost before running it |

Exercises: `p0-hello`, `p1-pointers`, `p2-indexing`, `p3-floats`, `p4-math`,
`p5-estimate` — all CPU-only, all self-checking.

[**There is a nine-question test**](prerequisites/README.md#should-you-skip-this)
at the top of that part. Pass it confidently and skip straight to Chapter 00.

### Part 0 - The machine

| # | Chapter | What you walk away with |
|---|---|---|
| 00 | [Why GPUs exist](chapters/00-why-gpus-exist.md) | Latency-hiding vs latency-reducing; why a GPU is a throughput machine and what that costs you |
| 01 | [The cost model: roofline](chapters/01-the-cost-model.md) | Arithmetic intensity, the roofline, Little's Law; predicting a kernel's runtime on paper |

Exercises: `00-hello-gpu` (probe the machine, derive its roofline knee).

### Part I - Foundations

| # | Chapter | What you walk away with |
|---|---|---|
| 02 | [Your first kernel](chapters/02-first-kernel.md) | Grid/block/thread, indexing, bounds guards, memory transfer, error handling |
| 03 | [Memory hierarchy and coalescing](chapters/03-memory-hierarchy.md) | Sectors, cache lines, why stride kills you, how to count bytes a warp actually moves |
| 04 | [Occupancy, warps and registers](chapters/04-occupancy.md) | What limits residency, why more occupancy is not always better, register pressure and spilling |
| 05 | [Divergence and the warp](chapters/05-divergence.md) | SIMT execution, predication, independent thread scheduling, cost of a branch |

Exercises: `02-vector-add`, `02-saxpy`, `03-stride-study`, `03-transpose-naive`, `04-occupancy-sweep`, `05-divergence`.

### Part II - The core patterns

| # | Chapter | What you walk away with |
|---|---|---|
| 06 | [Shared memory and tiling](chapters/06-shared-memory-tiling.md) | The scratchpad, bank conflicts, padding and swizzling, `__syncthreads` discipline |
| 07 | [Reductions](chapters/07-reductions.md) | Tree reduction, warp shuffles, atomics, two-pass vs persistent, the seven-version ladder |
| 08 | [Scan (prefix sum)](chapters/08-scan.md) | Hillis-Steele vs Blelloch, work efficiency, decoupled look-back |
| 09 | [Histograms and atomics](chapters/09-histograms-atomics.md) | Contention, privatization, aggregation, when atomics are free and when they are catastrophic |
| 10 | [Fusion: softmax and LayerNorm](chapters/10-fusion-softmax-layernorm.md) | Why fusion is the main lever in ML, online softmax, Welford, one-pass normalization |

Exercises: `06-transpose`, `07-reduce`, `08-scan`, `09-histogram`, `10-softmax`, `10-layernorm`.

### Part III - Compute-bound: GEMM to Tensor Cores

| # | Chapter | What you walk away with |
|---|---|---|
| 11 | [GEMM I: from naive to tiled](chapters/11-gemm-i.md) | Arithmetic intensity by construction, shared-memory tiling, register tiling, thread coarsening |
| 12 | [GEMM II: closing on cuBLAS](chapters/12-gemm-ii.md) | Vectorized loads, double buffering, conflict-free layouts, tail handling, autotuning |
| 13 | [Tensor Cores](chapters/13-tensor-cores.md) | `wmma` and `mma.sync`, fragments, `ldmatrix`, fp16/bf16/tf32 accumulation, CUTLASS orientation |
| 14 | [Async pipelines: cp.async, TMA, warp specialization](chapters/14-async-pipelines.md) | Overlapping global loads with math; producer/consumer warps; Hopper's hardware copy engine |

Exercises: `11-gemm-ladder` (five versions), `13-tensor-core-gemm`, `14-pipelined-gemm`.

### Part IV - Engineering rigor

| # | Chapter | What you walk away with |
|---|---|---|
| 15 | [Numerics and precision](chapters/15-numerics.md) | fp32/tf32/bf16/fp16/fp8, why you accumulate wide, catastrophic cancellation, determinism |
| 16 | [Profiling: Nsight and SASS](chapters/16-profiling.md) | Speed-of-light analysis, stall-reason taxonomy, reading SASS, a diagnostic decision tree |
| 17 | [Streams, concurrency and CUDA Graphs](chapters/17-streams-concurrency.md) | Overlapping copy and compute, launch overhead, graphs, why small kernels are CPU-bound |
| 18 | [The memory model and races](chapters/18-memory-model.md) | `__syncthreads` hazards, fences, volatile vs atomic, cooperative groups, `compute-sanitizer` |

Exercises: `15-precision-lab`, `16-diagnose` (three broken kernels, profiler only), `17-overlap`, `18-race-hunt`.

### Part V - Real ML kernels

| # | Chapter | What you walk away with |
|---|---|---|
| 19 | [Fused attention to FlashAttention](chapters/19-flash-attention.md) | Deriving FlashAttention from the roofline: online softmax, tiling, no materialized N x N |
| 20 | [Backward passes](chapters/20-backward-passes.md) | dSoftmax, dLayerNorm, flash backward, recompute vs store, atomics in gradients |
| 21 | [Quantized kernels](chapters/21-quantization.md) | int8/int4 packing, dequant GEMV for decode, why LLM inference is bandwidth-bound |
| 22 | [Sparsity, MoE and ragged batching](chapters/22-sparse-moe.md) | Gather/scatter, token routing, load imbalance, variable-length sequences |

Exercises: `19-flash-fwd`, `20-flash-bwd`, `21-int4-gemv`, `22-moe-dispatch`.

### Part VI - Ecosystem and mastery

| # | Chapter | What you walk away with |
|---|---|---|
| 23 | [Triton](chapters/23-triton.md) | The block-level programming model, autotuning, what Triton gives up, when to drop to CUDA |
| 24 | [Shipping: PyTorch integration and honest benchmarking](chapters/24-shipping.md) | Custom ops, `torch.compile`, CUDA graphs, benchmark methodology that survives review |
| 25 | [Multi-GPU](chapters/25-multi-gpu.md) | NCCL collectives, ring vs tree, overlapping communication with compute |
| -- | [Capstone](chapters/26-capstone.md) | Ship a fused kernel that beats PyTorch eager end-to-end, profiled and written up |

Exercises: `23-triton-ladder`, `24-torch-extension`, `25-allreduce`, `26-capstone`.

---

## Progress checklist

Tick these off as the gate prints `PASSED`. The performance bars are fractions
of a ceiling measured on your own GPU (`bench::copy_ceiling_gbs()` for
memory-bound work, cuBLAS or PyTorch for compute-bound work).

| Exercise | Correctness | Performance bar |
|---|---|---|
| `p0-hello` .. `p5-estimate` | all checks pass | laptop only, optional |
| `00-hello-gpu` | device report runs | -- |
| `02-vector-add` | exact | >= 80% of copy ceiling |
| `02-saxpy` | 1e-5 | >= 80% of copy ceiling |
| `03-stride-study` | exact | explain the curve, no bar |
| `03-transpose-naive` | exact | measure and account for the loss |
| `04-occupancy-sweep` | exact | find the config, explain the peak |
| `05-divergence` | exact | explain cost per divergent branch |
| `06-transpose` | exact | >= 85% of copy ceiling |
| `07-reduce` | 1e-5 | >= 85% of copy ceiling |
| `08-scan` | 1e-5 | >= 60% of copy ceiling |
| `09-histogram` | exact | >= 50% of copy ceiling |
| `10-softmax` | 1e-5 | >= 85% of copy ceiling |
| `10-layernorm` | 1e-4 | >= 85% of copy ceiling |
| `11-gemm-ladder` v1..v5 | 1e-3 | v5 >= 70% of cuBLAS fp32 |
| `13-tensor-core-gemm` | 2e-2 (fp16) | >= 50% of cuBLAS fp16 |
| `14-pipelined-gemm` | 2e-2 | beat your own v5 by >= 1.15x |
| `15-precision-lab` | -- | explain each failure |
| `16-diagnose` | fix all three | each >= its stated bar |
| `17-overlap` | exact | >= 1.6x over the serial version |
| `18-race-hunt` | sanitizer clean | -- |
| `19-flash-fwd` | 2e-3 (O and L) | >= 2.5x the 3-kernel cuBLAS baseline, O(N) workspace |
| `20-flash-bwd` | 3e-3 (dQ, dK, dV) | >= 2.0x the unfused cuBLAS backward |
| `21-int4-gemv` | 2e-2 | >= 80% of copy ceiling on packed weights |
| `22-moe-dispatch` | 2e-2 | >= 2.8x dense-masked (70% of the ideal 4x) |
| `23-triton-ladder` | matches your CUDA | >= 80% of your CUDA version |
| `24-torch-extension` | matches eager | >= 1.3x over eager, benchmarked honestly |
| `25-allreduce` | exact | >= 60% of NCCL |
| `26-capstone` | end-to-end correct | your own stated bar, defended |

---

## When you are stuck

There is a right way to be stuck, and it is not "look it up". Work this list in
order; it is the same list you will use for the rest of your career.

1. **Is it correct?** Run `make sanitize`. Out-of-bounds writes and races
   produce "wrong sometimes", which people misdiagnose as "wrong algorithm" for
   hours.
2. **What did you expect?** Re-derive the byte count and FLOP count by hand.
   Most "mystery" slowness is a kernel moving 3x more bytes than you believed.
3. **Where is the time going?** `make profile`. Look at the Speed Of Light
   section first: is this kernel memory-bound, compute-bound, or neither (which
   means latency-bound, and means occupancy or dependencies)?
4. **What does one warp do?** Take thread 0 of warp 0 and trace its addresses
   for one loop iteration on paper. Then thread 31. If those two addresses are
   far apart, you have found your problem.
5. **Shrink it.** 64 elements, one block, `printf` from thread 0. Nearly every
   indexing bug is visible at N=64 and invisible at N=1e7.
6. **Change one thing.** Measure. If the number moves and you cannot say why,
   you have not learned anything -- put it back and find out why first.

Each chapter ends with a "common failure modes" section written as symptoms,
not solutions. Use it as a differential diagnosis, not a lookup table.

---

## Reference material

- [Glossary](reference/glossary.md) - every term, defined once, properly.
- [Hardware cheat sheet](reference/hardware-cheatsheet.md) - the numbers for T4
  through H100, and the latencies you should have memorised.
- [Profiling cheat sheet](reference/profiling-cheatsheet.md) - which counter
  answers which question.
- [Reading list](reference/reading-list.md) - the papers, docs and codebases
  worth your time, in the order they become useful.

## Prerequisites

C or C++ at the level of "I can read a pointer and a 2D array flattened into
1D". No prior parallel programming. Comfort with the idea of a matrix multiply.
That is all. Everything about the GPU is built from zero here.

**If you do not have that yet, you are not stuck** —
[Part -1](prerequisites/README.md) teaches it from nothing, on your own
laptop, with no GPU required. Take the
[nine-question test](prerequisites/README.md#should-you-skip-this) and find
out which half of this paragraph you are in. Six to ten hours of work there
will save you weeks of confusion here.
