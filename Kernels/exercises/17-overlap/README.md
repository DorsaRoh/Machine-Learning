# 17 - copy-compute overlap

**Chapter:** [17](../../chapters/17-streams-concurrency.md)

## Task

Process 256 MB of host data in chunks: copy in, compute, copy out. Build a
pipelined version that beats the serial one by **1.6x**.

## Predict first

Measure the three phases separately first (time one chunk's H2D, one kernel,
one D2H), then predict:

```
per chunk: H2D ______ ms   compute ______ ms   D2H ______ ms
serial total for N chunks  = ______ ms
bottleneck stage           = ______
theoretical pipelined total= ______ ms
predicted speedup          = ______x
best possible speedup      = ______x   (what sets this bound?)
```

If your compute time is much smaller than your copy time, the best possible
speedup is small no matter how good your schedule is. Adjust `work` until the
stages are comparable -- and notice that doing so is itself the insight: a
pipeline only helps when the stages are balanced.

## Run

```bash
make run
make run ARGS="67108864 2097152 64 4"     # total, chunk, work, streams
```

## Pass bar

- Correctness: pipelined output identical to serial, 1e-6.
- Performance: >= 1.6x over serial.

## The four things that silently kill overlap

Check all of them before debugging anything else:

1. **Pageable host memory.** The harness pins it for you, so this one is
   handled -- but know that `cudaMemcpyAsync` from pageable memory is
   synchronous, and this is the most common cause of "my pipeline does
   nothing".
2. **The default stream.** One launch or copy without an explicit stream
   serialises everything around it.
3. **Shared device buffers.** If all streams write into the same staging
   buffer, chunk k+1's copy overwrites chunk k's data mid-kernel. You will see
   wrong results, not slow ones.
4. **Synchronising per chunk.** `cudaStreamSynchronize` inside the loop turns
   your pipeline back into the serial version.

## Verify with the timeline, not the number

```bash
nsys profile -o timeline --trace=cuda ./exercise
nsys stats timeline.nsys-rep
```

You should see three rows of activity -- H2D, kernel, D2H -- overlapping in
time. If you see them strictly sequential but the number improved anyway, find
out what actually got faster.

## The sweeps

1. **Chunk size.** From 64 KB to 64 MB. Plot speedup. There is an optimum and
   both sides of it have different causes: too small means per-chunk overhead
   dominates, too large means not enough chunks to fill the pipeline. Identify
   which side you are on at each end.
2. **Stream count.** 1, 2, 3, 4, 8, 16 at the best chunk size. Where does it
   stop helping, and why that number? (Hint: how many copy engines does your
   GPU have? `cudaDeviceProp::asyncEngineCount`.)
3. **Work per element.** 8, 64, 512. At which value does compute become the
   bottleneck, and what happens to the achievable speedup on either side?
4. **Deliberately break it.** Switch to `cudaMalloc`'d pageable host memory and
   re-measure. Confirm the overlap disappears in the timeline. Doing this once
   on purpose makes you recognise it instantly in the wild.

## After it passes

- Add `cudaEvent` dependencies so the D2H of chunk k waits on the kernel of
  chunk k rather than relying on stream ordering. Does anything change? Should
  it?
- Try `cudaMallocAsync` with a memory pool instead of pre-allocated buffers.
  Does it synchronise?
- Wrap the whole pipeline in a CUDA Graph. Does it help here? Why or why not?
  (Chapter 17 section 4: what has to be fixed for a graph, and is it here?)

<details>
<summary>Hints</summary>

- Which operations must be ordered with respect to each other, and which must
  not? Draw the dependency graph for 3 chunks before writing the loop.
- `n_streams` streams and `chunks` chunks: which stream does chunk k use, and
  what guarantees that stream is free?
- The device buffer is `n_streams * chunk * 2` floats. What is the offset for
  stream `s`'s input region? Its output region?
- Where does the single synchronisation go?

</details>
