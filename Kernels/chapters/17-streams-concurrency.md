# Chapter 17 - Streams, concurrency and CUDA Graphs

> Every chapter so far optimised a single kernel. This one is about the gaps
> *between* kernels, which in a real model can be a third of the runtime and
> which no amount of kernel tuning will touch.

---

## 1. Streams

A **stream** is an ordered queue of GPU work. Operations in the same stream
execute in order; operations in different streams may overlap.

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

kernel_a<<<g, b, 0, s1>>>(...);   // these two may run
kernel_b<<<g, b, 0, s2>>>(...);   // concurrently

cudaStreamSynchronize(s1);
```

### The null stream

Launches with no stream go to the **default (null) stream**, which has legacy
synchronising semantics: it implicitly synchronises with every other blocking
stream. One accidental null-stream launch in the middle of a concurrent
pipeline serialises everything, and this is a classic and very annoying bug.

Compile with `--default-stream per-thread`, or create streams with
`cudaStreamNonBlocking`, to avoid it.

### What can actually overlap

- Kernel with kernel (if resources allow -- two kernels that each fill the GPU
  will not overlap, they will just interleave at block granularity).
- Kernel with host-device copy, **if the host memory is pinned**.
- Copy in each direction simultaneously (separate hardware copy engines).
- Host code with everything on the device.

**Pinned (page-locked) memory is required for async copies:**

```cuda
cudaMallocHost(&h_buf, bytes);                 // pinned
cudaMemcpyAsync(d, h_buf, bytes, kind, stream);
```

With pageable memory, `cudaMemcpyAsync` silently falls back to a synchronous
staged copy and all your overlap disappears. This is the single most common
reason a carefully-built pipeline shows no speedup.

---

## 2. The copy-compute pipeline

The canonical use: a dataset too large to fit, processed in chunks.

```
serial:
  [copy 0][compute 0][copy 0 back][copy 1][compute 1][copy 1 back]...

pipelined (3 streams):
  copy:     [c0][c1][c2][c3]
  compute:      [k0][k1][k2][k3]
  copyback:          [b0][b1][b2]
```

Speedup approaches `min(n_stages, total/bottleneck_stage)`. With copy and
compute roughly balanced, 2x; with three overlapping phases, up to 3x. The
limit is the slowest stage -- if the copy takes twice as long as the compute,
you cannot do better than the copy time.

```cuda
for (int i = 0; i < chunks; ++i) {
    cudaStream_t s = streams[i % n_streams];
    cudaMemcpyAsync(d_in + off, h_in + off, sz, H2D, s);
    kernel<<<g, b, 0, s>>>(d_in + off, d_out + off, n);
    cudaMemcpyAsync(h_out + off, d_out + off, sz, D2H, s);
}
cudaDeviceSynchronize();
```

Within a stream the three operations are ordered (which you need for
correctness); across streams they overlap (which is the point).

---

## 3. Events

```cuda
cudaEvent_t e;
cudaEventCreate(&e);
cudaEventRecord(e, s1);
cudaStreamWaitEvent(s2, e, 0);   // s2 waits for s1 to reach e
```

Events give you a dependency graph across streams, for cases where stage B in
one stream needs stage A from another. They are also what you time with
(`cudaEventElapsedTime`) -- which `bench.cuh` already does.

`cudaEventSynchronize` blocks the host. `cudaStreamWaitEvent` blocks only a
stream, on the device, with no host involvement -- prefer it.

---

## 4. Launch overhead and CUDA Graphs

Each kernel launch costs 3-10 microseconds of CPU time. For a kernel that runs
for 500 microseconds, irrelevant. For one that runs for 3 microseconds, you are
spending more time asking than doing.

This is the dominant problem in **LLM inference decode**: generating one token
runs hundreds of small kernels, each a few microseconds, and the CPU cannot
issue them fast enough. The GPU idles waiting to be told what to do.

**CUDA Graphs** record a sequence of launches once and replay it as a single
submission:

```cuda
cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
for (int i = 0; i < 100; ++i) kernel<<<g, b, 0, s>>>(...);
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

for (int iter = 0; iter < many; ++iter)
    cudaGraphLaunch(exec, s);   // one launch for all 100 kernels
```

The overhead moves from `n * 5us` to about `5us` total. For a decode step with
200 small kernels, that is 1 ms per token of pure overhead removed, which can
be most of the step.

**The constraint:** the graph is fixed. Same kernels, same arguments, same
shapes, every replay. Pointers must be stable, so you need pre-allocated
buffers. Control flow that depends on data cannot be captured. This is exactly
why inference engines use fixed-size KV cache buffers and padded batch shapes:
not for elegance, but to make the work graph-capturable.

`cudaGraphExecUpdate` lets you change some parameters without re-instantiating,
which covers the common "same graph, different pointers" case.

---

## 5. Concurrency between kernels

Two kernels in different streams overlap only if the GPU has resources for
both. A kernel launching 10,000 blocks fills the machine; a second kernel's
blocks simply queue behind it. Overlap is real when:

- Both kernels are small (few blocks).
- They use different resources (one memory-bound, one compute-bound).
- You deliberately limit each kernel's grid.

**Stream priorities:**

```cuda
cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, priority);
```

Higher-priority streams get their blocks scheduled first. Useful for latency-
sensitive work sharing a GPU with throughput work -- an inference request
alongside background batch processing, for example.

**MPS** (Multi-Process Service) lets multiple *processes* share a GPU with real
concurrency rather than time-slicing. Relevant for serving many small models.

**Green contexts / partitioning** (recent CUDA) carve the SMs into disjoint
sets, giving hard isolation instead of best-effort sharing.

---

## 6. Synchronisation, ranked by cost

```cuda
cudaDeviceSynchronize();      // everything. Avoid in hot paths.
cudaStreamSynchronize(s);     // one stream.
cudaEventSynchronize(e);      // one point.
cudaStreamWaitEvent(s2, e);   // device-side, no host involvement. Cheapest.
```

Implicit synchronisation points that catch people out:

- `cudaMalloc` / `cudaFree` (device-wide sync)
- Pageable `cudaMemcpy`
- A null-stream launch among non-blocking streams
- `cudaMemset` on the default stream

**Never allocate in a hot loop.** `cudaMalloc` synchronises the device and
costs tens of microseconds. Pre-allocate, or use a pool
(`cudaMallocAsync` with a memory pool is the modern option and does not
synchronise).

---

## 7. Common failure modes

**Streams show no overlap in the timeline.** Pageable host memory, or a
null-stream operation in between, or the first kernel already fills the GPU.

**`cudaMemcpyAsync` behaves synchronously.** Not pinned.

**Pipelined version slower than serial.** Chunks too small, so per-chunk
overhead dominates. Or too many streams thrashing.

**Graph capture fails.** An unsupported operation inside the capture --
synchronisation, an allocation, or a host callback.

**Graph produces stale results.** Captured pointers no longer point at the
current data. Graphs capture *values*, including pointer values.

**Everything is slow and the GPU is 40% idle.** Launch-bound. Count kernels
per iteration and their average duration; if duration is near 5 microseconds,
you have found it.

**Out of memory after adding streams.** Each stream needs its own buffers. n
streams, n times the staging memory.

---

## 8. Exercises

### 17.1 - Pipeline arithmetic

A 4 GB dataset, chunked, with per-chunk copy-in 10 ms, compute 8 ms, copy-out
10 ms.

(a) Serial total for 10 chunks.
(b) With 2 streams?
(c) With 3? With 4?
(d) Theoretical minimum, and which stage sets it?
(e) At what compute time per chunk does compute become the bottleneck?
(f) If H2D and D2H share one copy engine instead of two, redo (b)-(d).

### 17.2 - Launch overhead

(a) Time an empty kernel launched 10,000 times. Per-launch cost?
(b) The same inside a CUDA Graph. Per-launch cost now?
(c) At what kernel duration does graph capture stop being worth it?
(d) A decode step with 200 kernels averaging 4 us of work: total time with and
    without graphs. What fraction of the step was overhead?
(e) What would you have to change about the model code to make the step
    capturable?

### 17.3 - Build the pipeline

In `17-overlap`, process a dataset larger than GPU memory with a 3-stage
pipeline.

(a) Predict the speedup over serial.
(b) Implement with pinned memory and n streams.
(c) Sweep chunk size. Plot. There is an optimum -- explain both sides of it.
(d) Sweep stream count at the best chunk size.
(e) Deliberately switch to pageable memory. Measure. Confirm the overlap
    disappears, and confirm it in the `nsys` timeline rather than just from the
    number.

### 17.4 - When do kernels overlap?

(a) Launch two kernels, each with 100,000 blocks, in two streams. Overlap?
(b) Two kernels with 4 blocks each. Overlap?
(c) One memory-bound and one compute-bound kernel, each half-filling the GPU.
    Does the total time equal the max, the sum, or something in between?
(d) Verify each with `nsys`.
(e) What does this imply about whether splitting one big kernel into two
    streams can ever help?

### 17.5 - Graph a real sequence

Take your fused softmax plus LayerNorm plus an elementwise op, run 1000 times.

(a) Time without graphs.
(b) Capture and replay. Time.
(c) Speedup, and what fraction of the original was launch overhead?
(d) Change an input pointer and re-launch the graph without updating it. What
    happens? (Do this deliberately once; it is instructive.)
(e) Use `cudaGraphExecUpdate` to change it properly.

---

## Exercises to run

- [`17-overlap`](../exercises/17-overlap) - bar: 1.6x over the serial version.

---

## What you should now believe

- Streams give ordering within and concurrency across; the null stream
  serialises everything.
- Async copies require pinned memory, or they silently are not async.
- A copy-compute pipeline is bounded by its slowest stage.
- Launch overhead is 3-10 us and dominates small-kernel workloads; CUDA Graphs
  remove it at the cost of requiring fixed shapes.
- `cudaMalloc` synchronises; never allocate in a hot loop.
- Kernel-kernel overlap only happens when the GPU has spare capacity.

Next: [Chapter 18 - The memory model and races](18-memory-model.md).
