# Capstone

> Everything until now had a gate that told you when you were done. This one
> does not. You choose the problem, you set the bar, and you defend both. That
> is the actual job.

---

## The brief

**Ship a fused CUDA kernel that makes a real model measurably faster
end-to-end, and write it up so that a skeptical reviewer believes you.**

Three requirements that do the work of a gate:

1. **Real model, end-to-end.** Not a microbenchmark. The number you report is
   tokens/second or step time for a model someone would actually run.
2. **An honest baseline.** `torch.compile` on the same hardware, same dtype,
   same shapes. Not eager, unless you can argue eager is what ships.
3. **A write-up that includes where it does not help.**

---

## Choosing a project

Pick one. The first three are the safest choices if you want a result in a
week; the last three are the ones that will teach you the most.

**A. Fused transformer block epilogue.**
RMSNorm + SwiGLU + residual, fused into one kernel. Tests everything in
Chapters 06, 10 and 15. Memory-bound, so the ceiling is computable in advance
and you will know exactly how close you are. The most reliable way to get a
real end-to-end win, because these ops are everywhere and individually cheap
enough that launch overhead and traffic dominate.

**B. Attention variant that does not exist yet in a fast form.**
Sliding window with a twist, ALiBi or a custom positional scheme, attention
with a learned sparsity pattern, or a masking rule your research needs.
Builds directly on `19-flash-fwd`. The win is real because the alternative is
materializing the mask.

**C. Quantized inference kernel.**
Your `21-int4-gemv` extended: a full W4A16 linear layer with bias and
activation fused, integrated into a real 7B model, measured in tokens/second
against the fp16 model. Bandwidth-bound, with a hard ceiling you can quote.

**D. Fused optimizer.**
AdamW over all parameters in one kernel, multi-tensor, with the master-weight
and loss-scaling logic from Chapter 20. Tests Chapters 03, 15 and 17. The win
comes from launch overhead and traffic, and it applies to every model.

**E. Backward of something fused.**
Take A or B and ship the backward too, with `gradcheck` and a training run that
converges. Substantially harder than the forward and much more valuable.

**F. Your own.**
If you have a workload with a kernel-shaped hole in it, that is the best
project. Write the one-paragraph justification first: what is slow, why, and
what the roofline says is available.

---

## Deliverables

### 1. The kernel

- CUDA (or Triton, with a justification for the choice).
- Correct: `compute-sanitizer` clean under memcheck and racecheck, on the real
  workload, not a toy input.
- Tested: a test suite covering edge shapes, non-contiguous inputs, dtypes you
  claim to support, and the degenerate cases (size 0, size 1, size not a
  multiple of your tile).
- If it has a backward: `torch.autograd.gradcheck` in float64.

### 2. The integration

- A `torch.library.custom_op` with a registered fake implementation.
- Traces through `torch.compile` with no graph break (`TORCH_LOGS=graph_breaks`
  as evidence).
- Uses the current stream and the caching allocator.
- Builds as a proper extension, for at least two architectures plus a PTX
  fallback.

### 3. The analysis

This is the part that distinguishes a kernel engineer from someone who got a
kernel to work.

- **The roofline, before you wrote anything.** Bytes, FLOPs, arithmetic
  intensity, which side of the ridge, and the predicted time.
- **Nsight Compute output** for your final kernel: Speed-of-Light, the
  dominant stall reason, achieved bandwidth or achieved TFLOP/s as a fraction
  of the ceiling.
- **The optimization history.** Each version, what you changed, what you
  predicted, what you measured, and where you were wrong. The wrong predictions
  are the most interesting rows in the table.
- **The remaining gap.** You are at X% of the ceiling. Account for the missing
  (100-X)%. "I do not know" is a worse answer than a wrong hypothesis with a
  measurement attached.

### 4. The benchmark

To the Chapter 24 methodology, with no exceptions:

- Hardware, driver, CUDA, PyTorch versions.
- Shapes and dtypes, and why those.
- Baseline, and the argument for why it is the right one.
- Warmup, iteration count, timing method, cache state, clock state.
- Median and spread, kernel-level *and* end-to-end.
- Accuracy against the baseline, in the same table as the speed.
- **Where it does not help.** Shapes, dtypes or hardware where your kernel is
  neutral or worse. Every kernel has them; a write-up without this section is
  not believable.

### 5. The write-up

2000 words or fewer. Structure that works:

1. What is slow, and the roofline that proves it.
2. What you built, and the one or two ideas it rests on.
3. Results, with the methodology.
4. What you tried that did not work, and why.
5. What is left on the table, and what you would do next.

---

## Rubric

Score yourself honestly. 70 is a pass; below 55 means pick a smaller project
and finish it properly rather than patching this one.

| | Points | What full marks looks like |
|---|---|---|
| Correctness | 20 | Sanitizer clean on the real workload; edge cases tested; gradcheck if applicable |
| Performance | 20 | A real end-to-end win against an honest baseline, with the kernel near its computed ceiling |
| Analysis | 25 | Roofline before and after; profiler evidence; the remaining gap accounted for |
| Benchmark honesty | 20 | Full methodology; a "where it does not help" section with real content |
| Integration | 10 | Custom op, fake impl, no graph break, builds for multiple architectures |
| Write-up | 5 | A reviewer can reproduce your result from it |

The weighting is deliberate: **analysis and benchmark honesty together are 45
points, more than correctness and performance combined.** A kernel that is 1.3x
faster with a rigorous account of why is worth more than one that is 3x faster
with no explanation, because only the first one teaches you how to get the next
one.

---

## A suggested shape for the work

- **Day 1.** Profile the model. Find the real bottleneck (Chapter 24, exercise
  24.3). Compute the roofline. Decide what is available. Write down the
  predicted speedup *before* you write code.
- **Day 2.** The simplest correct version. Test harness, oracle, sanitizer.
- **Days 3-4.** Optimize, one change at a time, predicting before each and
  recording the table.
- **Day 5.** Integration, `torch.compile`, extension build.
- **Day 6.** Benchmark properly. Measure end-to-end. Find out that it is less
  than your microbenchmark suggested, and find out why.
- **Day 7.** Write it up. Include day 6's disappointment.

---

## Are you actually done?

Not with the capstone -- with the course. You should be able to do all of these
without looking anything up:

**Predict.**
- [ ] Given a kernel, state within 2x how long it should take, and why.
- [ ] Place any operation on the roofline from its shapes alone.
- [ ] Say which of bandwidth, compute, latency or occupancy limits it.
- [ ] Predict the effect of a change before making it, and be right most of
      the time.

**Diagnose.**
- [ ] Read an Nsight Compute report and name the bottleneck in under a minute.
- [ ] Recognize uncoalesced access, bank conflicts, spills, divergence and
      atomic contention from their counter signatures.
- [ ] Read SASS well enough to confirm tensor cores, async copies and spills.
- [ ] Find a race with `compute-sanitizer` and explain why it was silent.

**Build.**
- [ ] Write a coalesced, correct kernel for a new problem on the first try.
- [ ] Reach 80%+ of the ceiling on anything memory-bound.
- [ ] Reach 70%+ of cuBLAS on a GEMM you wrote.
- [ ] Fuse a chain of ops and know in advance what fusing will buy.
- [ ] Derive and implement a backward pass.

**Judge.**
- [ ] Decide whether a kernel is worth writing before writing it.
- [ ] Choose between CUDA and Triton for a given task, with reasons.
- [ ] Design a benchmark nobody can poke a hole in.
- [ ] Say honestly when something is already fast enough.

The last item on that list is the hardest one and the most valuable.

---

## Where to go from here

- **Read CUTLASS**, properly. You now have every concept it uses. Start with
  the CuTe layout algebra -- it is the formalization of everything Chapters 12
  to 14 did informally.
- **Read FlashAttention-3**, and then its git history. Watching a
  state-of-the-art kernel evolve is the closest thing to apprenticeship.
- **Contribute.** PyTorch, vLLM, SGLang, Triton, llama.cpp all have open kernel
  issues. A profiled, benchmarked PR against a real one is worth more on your
  résumé than any course certificate.
- **Pick a hard thing and go deep.** Hopper's TMA and warp specialization.
  Blackwell. Distributed GEMM. fp4. Persistent kernels for serving. The field
  moves, and the way to stay current is to have gone deep once.
- **Teach it.** Write the blog post explaining the thing that took you three
  days to understand. It is the fastest way to find out what you still do not
  know.

---

## What you should now believe

- A kernel is worth writing only when the roofline says there is something to
  win, and you can tell before you start.
- Prediction is the skill. Everything else -- profiling, SASS, the instruction
  set -- exists to tell you why your prediction was wrong.
- Honest measurement is a technical skill with a method, and most people do not
  have it.
- You are not done learning this; you are done needing to be taught it.

Back to the [syllabus](../README.md).
