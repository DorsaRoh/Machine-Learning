# Chapter 24 - Shipping: PyTorch integration and honest benchmarking

> A kernel that is 3x faster in your microbenchmark and 1.02x faster in the
> model is the normal outcome, not a rare one. This chapter is about the gap
> between those two numbers: how to close it, and how to measure it in a way
> that survives someone else trying to reproduce it.

---

## 1. Getting the kernel into PyTorch

Three levels, in increasing order of commitment.

**Inline, for iteration.**

```python
from torch.utils.cpp_extension import load_inline
mod = load_inline(name="mykernel",
                  cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["my_op"], extra_cuda_cflags=["-O3", "--use_fast_math"])
```

Compiles on first call, caches in `~/.cache/torch_extensions`. Perfect for the
development loop, unacceptable for a package.

**A built extension**, with `setup.py` and `CUDAExtension`, compiled ahead of
time and installed. This is what a library ships.

**A `torch.library` custom op**, which is what makes it a first-class citizen:

```python
@torch.library.custom_op("mylib::fused_norm", mutates_args=())
def fused_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return mod.fused_norm(x, w, eps)

@fused_norm.register_fake
def _(x, w, eps):
    return torch.empty_like(x)
```

The `register_fake` ("meta") function computes output shapes and dtypes without
touching data. Without it, `torch.compile` cannot trace through your op and
will break the graph around it -- which can cost more than your kernel saves.
This is the single most commonly skipped step and the most expensive one to
skip.

For a trainable op, wrap it in `torch.autograd.Function` with `forward` and
`backward`, or register an autograd formula for the custom op. Chapter 20 is
the content of `backward`; this is the plumbing.

Rules that are easy to get wrong:

- **Use the current stream.** `at::cuda::getCurrentCUDAStream()`. Launching on
  the default stream in a custom op will silently serialize against everything
  else or, worse, race with it.
- **Declare mutation.** If you write into an input, say so in `mutates_args`.
  The compiler reorders around ops it believes are pure.
- **Check contiguity and dtype at the boundary**, with `TORCH_CHECK`. A
  non-contiguous input that your kernel assumes is contiguous produces silent
  garbage, and PyTorch hands you non-contiguous tensors constantly.
- **Allocate with `torch.empty`,** not `cudaMalloc`. The caching allocator is
  stream-aware and `cudaMalloc` is a synchronizing call.

---

## 2. Benchmarking that survives review

This section is the reason the chapter exists. Most published kernel speedups
are wrong, and they are wrong in a small number of specific, repeatable ways.

### Warm up properly

The first call pays for CUDA context creation, JIT, autotuning, allocator
growth and cuDNN/cuBLAS algorithm selection. The first few calls after that run
at a lower clock because the GPU was idle. Discard at least 10 iterations, more
if you are autotuning.

### Time with events and synchronize

```python
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters): fn()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / iters
```

Wall-clock time around an async launch measures how fast your CPU can enqueue
work. It is the single most common benchmarking error and it makes tiny kernels
look infinitely fast.

### Decide what is in cache, deliberately

Running the same kernel on the same 8 MB buffer 100 times means every run after
the first reads from L2. That is a legitimate thing to measure if your real
workload is cache-resident -- and a lie otherwise. Flush L2 between iterations
(Chapter 03, `bench::L2Flusher`) or rotate through buffers larger than L2. Say
which you did.

### Lock the clocks, or report that you did not

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
sudo nvidia-smi -lgc 1200,1200     # pin the graphics clock
```

Boost clocks drift with temperature. A 10% difference between the first and
twentieth run of a long benchmark is thermal, not algorithmic. On shared or
cloud GPUs you often cannot lock clocks; then interleave your kernel and the
baseline rather than running all of one then all of the other, so they see the
same thermal conditions.

### Report the median, and the spread

Means are dragged by the occasional 10 ms outlier from an unrelated process.
Report median and, at minimum, min and p95. If your p95 is 3x your median,
something is wrong and the median is hiding it.

### Compare against a real baseline

The dishonest comparisons, in order of frequency:

- Your fused fp16 kernel against unfused fp32 eager.
- Your kernel against a baseline you wrote, badly, yesterday.
- Your kernel at its best shape against the baseline at its worst.
- Your kernel with `--use_fast_math` against a baseline without.
- A kernel-only time against a baseline measured end-to-end, or vice versa.
- Batch size 1 for yours, batch size 1 for theirs, when the deployment is 64.

The baseline is the best thing a competent person would actually use. If that
is `torch.compile`'s generated kernel, benchmark against that.

### Report accuracy alongside speed

A 2x speedup at 1e-2 error against a baseline at 1e-6 is not a speedup, it is a
different computation. State the tolerance and the dtype in the same table as
the time.

### Measure the model, not just the kernel

A kernel that is 40% of a 5%-of-total-runtime op gives you 2% end-to-end, and
Amdahl's law has no exceptions. Profile the model first, find what is actually
expensive, and quote the end-to-end number even when the kernel number is more
flattering.

---

## 3. Why the model-level speedup disappears

The usual suspects, each with a fix:

**Launch overhead.** 5-10 us per launch (Chapter 17). A model with 2000 small
kernels per step spends 10-20 ms just launching. CUDA Graphs or
`torch.compile`'s `mode="reduce-overhead"` capture the whole sequence and
replay it with one launch.

**Graph breaks.** Your custom op without a fake implementation forces
`torch.compile` back to eager around it, losing the fusion of neighbouring ops.
Check with `TORCH_LOGS=graph_breaks`.

**Memory format.** PyTorch hands you channels-last, or a non-contiguous view,
and your kernel calls `.contiguous()` -- which is a full copy you did not count
in your microbenchmark.

**The op was not the bottleneck.** Profile before optimizing. Every time.

**Allocator pressure.** Your kernel allocates a workspace per call; the caching
allocator handles it, until fragmentation makes it call `cudaMalloc` and
synchronize. Allocate once and reuse.

**CPU-bound.** At small batch sizes the Python and dispatcher overhead exceeds
the GPU time, and the GPU is idle between kernels. `nsys` shows this instantly:
gaps on the CUDA row. No kernel optimization fixes it.

---

## 4. Profiling at the model level

```python
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Then `nsys profile -o trace python train.py` for the timeline, which answers
the questions the table cannot: is the GPU idle, are the copies overlapping,
how big are the gaps between kernels.

The order of operations that works: **nsys first** (is the GPU busy?), **the
profiler table second** (which op?), **`ncu` third** (why is that kernel
slow?). Going straight to `ncu` means micro-optimizing a kernel that was never
the problem.

---

## 5. Packaging, briefly

- Compile for the architectures you support: `-gencode arch=compute_80,code=sm_80`
  for each, plus one `compute_XX,code=compute_XX` so newer GPUs can JIT from
  PTX. Omitting the PTX fallback is why extensions break on new hardware.
- Pin your CUDA toolkit version against the PyTorch build's. ABI mismatches
  produce link errors that look like missing symbols.
- Test on the oldest architecture you claim to support. `-arch=native` on your
  H100 and shipping to A100 users is a bug waiting.
- Put `compute-sanitizer` in CI. It catches the race that your tests do not.
- Version your kernel's numerics. A change that makes results bit-different is
  a breaking change for someone.

---

## 6. Common failure modes

**3x in the microbenchmark, 1.0x in the model.** Work section 3's list in
order.

**Faster than eager, slower than `torch.compile`.** Your baseline was wrong.
Re-benchmark.

**Results differ between the test and the model.** Non-contiguous input, or a
dtype you did not check for.

**Intermittent wrong results in training only.** A race that your single-stream
test never exercised. `compute-sanitizer` with the real workload.

**Benchmark numbers that move 20% between runs.** Clocks, or an unflushed
cache, or another process. Fix the methodology before believing any of it.

**Great on your A100, broken on the user's 4090.** Architecture flags, or an
assumption about shared memory size or warp scheduling.

---

## 7. Exercises

### 24.1 - Ship one kernel properly

Take your `10-layernorm`:

(a) Wrap it with `load_inline` and verify against `torch.nn.LayerNorm`.
(b) Convert to a `torch.library.custom_op` with a fake implementation.
(c) Add the backward (Chapter 20 section 3) via `autograd.Function`.
(d) `torch.autograd.gradcheck` in float64.
(e) Confirm `torch.compile` traces through it without a graph break.
(f) Build it as a proper extension with `setup.py` and `pip install -e .`.

### 24.2 - Break your own benchmark, then fix it

Write a benchmark of your kernel against eager that is wrong in each of these
ways, and record the speedup each one reports:

(a) No warmup.
(b) Wall-clock time with no synchronize.
(c) Same buffer every iteration, L2 hot.
(d) Mean instead of median, with one outlier.
(e) fp16 yours against fp32 baseline.
(f) Now fix all five. What is the honest number, and how far is it from the
    worst of the five? Keep this table; it is the most useful thing in this
    chapter.

### 24.3 - Find the real bottleneck

Take a small transformer (nanoGPT-scale) in PyTorch:

(a) Profile one training step. Rank ops by CUDA time.
(b) Which three ops are 50% of the time? Were they what you expected?
(c) For the top one, compute its roofline position. Is it near its ceiling?
(d) If you made it infinitely fast, what is the end-to-end speedup? (Amdahl.)
(e) Now do it for the second one. Which is actually worth your week?

### 24.4 - Measure the overhead floor

(a) Time an empty kernel launch, 10000 times. That is your per-launch floor.
(b) Count the kernel launches in one training step of your model.
(c) Multiply. What fraction of step time is launch overhead?
(d) Apply `torch.compile(mode="reduce-overhead")`. Re-measure.
(e) Apply CUDA Graphs manually. Compare.
(f) At which batch size does launch overhead stop mattering?

### 24.5 - The honest write-up

Write the benchmark section you would put in a PR for your best kernel:

(a) Hardware, driver, CUDA version, PyTorch version.
(b) Shapes, dtypes, and why those shapes.
(c) Baseline, and why it is the right baseline.
(d) Methodology: warmup, iterations, timing method, cache state, clock state.
(e) Median and spread, kernel-level and end-to-end.
(f) Accuracy against the baseline.
(g) Where it does *not* help, and what you did not test.

(g) is what distinguishes an engineer from a marketer, and reviewers notice its
absence.

---

## Exercises to run

- [`24-torch-extension`](../exercises/24-torch-extension) - one of your kernels
  as a real PyTorch op. Bar: matches eager, >= 1.3x end-to-end on the stated
  workload, benchmarked to the methodology above.

---

## What you should now believe

- A custom op without a fake implementation is a graph break, and a graph break
  can cost more than your kernel saves.
- Wall-clock timing around async launches is the most common benchmarking
  error; events and synchronize are the fix.
- Cache state, clock state, warmup, median-versus-mean and baseline choice each
  move the headline number by more than most kernel optimizations do.
- Microbenchmark speedups do not survive to the model for a small number of
  specific reasons, all of which you can check in an hour.
- `nsys` first, profiler table second, `ncu` third.
- Reporting where your kernel does not help is what makes the rest of your
  numbers believable.

Next: [Chapter 25 - Multi-GPU](25-multi-gpu.md).
