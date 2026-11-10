# Reading list

Ordered by when it becomes useful, not by importance. Reading CUTLASS before
you have written a tiled GEMM yourself is a way to feel productive without
learning anything.

## Read alongside Parts 0-II

- **CUDA C++ Programming Guide** -- not cover to cover. Read the chapters on
  the programming model, memory hierarchy, and the Performance Guidelines
  appendix. Return to it as a reference forever.
- **CUDA C++ Best Practices Guide** -- the memory optimisation chapter is the
  highest-density thing NVIDIA has written.
- **Mark Harris, "Optimizing Parallel Reduction in CUDA"** -- the seven-version
  slide deck that defined how this subject is taught. Chapter 07 makes you
  rediscover it rather than read it; read it *after* you finish the exercise.
- **"How to Access Global Memory Efficiently"** and the rest of the NVIDIA
  developer blog's CUDA Pro Tip series -- short, each one teaches exactly one
  thing.
- **Programming Massively Parallel Processors** (Kirk & Hwu) -- the textbook.
  Chapters 2-6 cover Parts I-II at a comfortable pace with different examples.

## Read alongside Part III (GEMM)

- **Scott Gray's maxas SGEMM writeup** -- hand-scheduled assembly GEMM. Dated
  hardware, undated thinking. The clearest explanation anywhere of why register
  tiling and double buffering are the whole game.
- **"Anatomy of High-Performance Matrix Multiplication"** (Goto & van de Geijn)
  -- the CPU paper that invented the blocking structure every GPU GEMM still
  uses.
- **CUTLASS documentation and `media/docs/efficient_gemm.md`** -- the canonical
  modern description of the GEMM hierarchy: threadblock, warp, instruction.
- **CuTe tutorial** (inside CUTLASS 3.x) -- layouts as a first-class algebra.
  Hard, and worth it once you have written a tiled GEMM by hand and felt the
  index arithmetic hurt.
- **Simon Boehm, "How to Optimize a CUDA Matmul Kernel"** -- a step-by-step
  ladder to ~95% of cuBLAS. Read it only after your own ladder stalls; it will
  hand you answers that Chapter 11 wants you to find.

## Read alongside Part IV

- **"Nsight Compute Kernel Profiling Guide"** -- the definitive mapping from
  metric names to meaning.
- **CUDA memory consistency model** (Programming Guide appendix) -- dense and
  worth it the day a race survives `__syncthreads()`.
- **"Demystifying GPU Microarchitecture through Microbenchmarking"** (Wong et
  al.) and the more recent Volta/Turing/Ampere microbenchmarking papers --
  how people learn the numbers that are not in any datasheet.
- **"What Every Computer Scientist Should Know About Floating-Point
  Arithmetic"** (Goldberg) -- for Chapter 15.

## Read alongside Part V

- **FlashAttention** (Dao et al., 2022) -- IO-awareness as a design principle.
  Read Section 3 with Chapter 19 open.
- **FlashAttention-2** (Dao, 2023) -- work partitioning: fewer non-matmul
  FLOPs, parallelism over sequence length, better warp layout.
- **FlashAttention-3** (Shah et al., 2024) -- Hopper: warp specialization, TMA,
  FP8, and overlapping softmax with matmul.
- **Online normalizer calculation for softmax** (Milakov & Gimelshein) -- two
  pages, and the mathematical core of FlashAttention.
- **"Efficiently Scaling Transformer Inference"** (Pope et al.) -- why decode
  is bandwidth-bound and what follows from that.
- **LLM.int8(), GPTQ, AWQ, Marlin** -- the quantized-kernel lineage for
  Chapter 21. Marlin's kernel is a masterclass in itself.
- **Megablocks / MegaScale MoE kernels** -- block-sparse and grouped GEMM for
  Chapter 22.

## Read alongside Part VI

- **Triton paper and tutorials** (Tillet et al.) -- the tutorials *are* the
  documentation; the fused-attention one is the reference implementation most
  people actually copy.
- **PyTorch custom C++/CUDA extensions tutorial** and the `torch.library`
  docs -- how to ship a kernel so it composes with autograd and `torch.compile`.
- **NCCL documentation** and the ring/tree allreduce explanations -- Chapter 25.

## Codebases worth reading (in this order)

1. **`cuda-samples`** -- small, focused, officially maintained.
2. **Triton's `python/tutorials/`** -- short, modern, real.
3. **FlashAttention repo** -- the gap between the paper and the code is where
   the real engineering lives.
4. **CUTLASS** -- the deep end. Templates obscure the ideas until suddenly they
   do not.
5. **vLLM's `csrc/`** -- production inference kernels: paged attention,
   quantized GEMMs, all with real-world constraints visible.
6. **llm.c** (Karpathy) -- a full transformer training loop in plain CUDA,
   optimised step by step with the history preserved in the commits. The single
   best "read the whole thing" project in this field.

## Places to keep up

- NVIDIA developer blog (the technical posts, not the launch posts)
- GTC session recordings -- the CUTLASS and Nsight deep dives especially
- The `gpu-mode` (formerly CUDA MODE) lecture series and its working group
- NVIDIA forums and the CUDA section of Stack Overflow, where the answers from
  Robert Crovella are effectively a second programming guide
