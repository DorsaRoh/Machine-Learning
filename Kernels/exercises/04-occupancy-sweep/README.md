# 04 - occupancy sweep

**Chapter:** [04](../../chapters/04-occupancy.md)

## Task

Write a grid-stride streaming kernel with a compile-time ILP factor. The
harness sweeps block size (32..1024), ILP (1..8) and blocks per SM (1..16) --
120 configurations -- and prints bandwidth next to the theoretical occupancy
reported by `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

## Predict first

```
best block size        = ______   why?
best ILP               = ______   why?
best blocks/SM         = ______   why?
peak bandwidth         = ______ GB/s
occupancy at the peak  = ______%
```

And the question the whole exercise exists for:

```
Will the highest-bandwidth configuration also be the highest-occupancy one?
  yes / no, because ____________________
```

## Run

```bash
make run
```

Takes a minute or two. Keep the table -- you will refer back to it.

## Pass bar

- Correctness: 1e-6, checked at ILP=4 with a grid that does not divide `n`
  evenly, so your tail handling is actually tested.
- Performance: the best configuration must reach >= 85% of the copy ceiling.

## The actual exercise

1. **Plot bandwidth against occupancy** for all 120 points. If occupancy
   predicted performance, this would be a line. Describe the shape you actually
   get, and name the points that break the relationship.

2. **Hold occupancy fixed, vary ILP.** Find two configurations with similar
   occupancy and different ILP. Which is faster? By how much? Explain via
   Little's Law: compute bytes-in-flight for each.

3. **Find the minimum occupancy that still reaches 90% of your peak.** This
   number is the honest answer to "how much occupancy do I need", and it is
   much lower than people assume.

4. **Explain the block=32 row.** It is bad. There are at least two independent
   reasons. Name both.

5. **Explain the blocks/SM=1 column.** Also bad. Different reason again.

6. **Register check.**
   ```bash
   nvcc -arch=native -Xptxas -v -c kernel.cu 2>&1 | grep -A2 stream_kernel
   ```
   Registers per thread at ILP=1 vs ILP=8. Does the increase explain the
   occupancy change the harness reports? Compute the occupancy by hand with the
   four-limiter formula from Chapter 04 and check it against the API.

7. **Find the knee.** At which ILP does adding more stop helping? Why there?
   Relate it to the number of outstanding memory requests a thread can have.

## Extension

Add a `BALLAST` template parameter that forces extra register usage (an array
of accumulators that genuinely survives to the end of the kernel so the
compiler cannot delete it). Sweep it. You can now move occupancy *independently
of ILP*, which cleanly separates the two effects people usually confuse.

<details>
<summary>Hints</summary>

- In one iteration of your loop, what addresses do the 32 lanes of a warp
  touch? If your thread handles `i, i+1, i+2, i+3`, how far apart are lane 0
  and lane 1?
- "Issue all loads, then use them" -- what does that look like in source? What
  does the compiler have to see to keep them independent?
- With ILP=8, how many registers hold loaded values simultaneously?
- `blocks/SM = 1` with `block = 256`: how many warps per SM? Is that enough to
  cover a 500-cycle latency at one load per warp?

</details>
