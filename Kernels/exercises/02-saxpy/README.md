# 02 - SAXPY

**Chapter:** [02](../../chapters/02-first-kernel.md)

## Task

`y[i] = alpha * x[i] + y[i]`, 64M fp32, at >= 80% of your measured ceiling.

## Predict first

This is the interesting part of the exercise.

```
vector add : reads a, reads b, writes c
saxpy      : reads x, reads y, writes y
```

Both touch three streams. So:

```
predicted traffic vs vector add  : more / less / same ?
predicted time vs vector add     : ______ ms
predicted FLOP count             : ______ (saxpy does 2 FLOPs per element, not 1)
predicted GFLOP/s                : ______
```

Now think harder about the write. In vector add, `c` is written but never read,
so the hardware must fetch its cache lines before modifying them
(write-allocate, Chapter 03 section 6) -- unless the write covers a whole line,
which a coalesced warp does. In SAXPY, `y` is read anyway, so the line is
already present.

**Does that make SAXPY faster than vector add, slower, or identical on your
GPU?** Commit to an answer, then measure both and explain whichever you got.

## Run

```bash
make run
```

## Pass bar

- Correctness: 1e-5.
- Performance: >= 80% of `copy_ceiling_gbs()`.

## After it passes

1. Measure both `02-vector-add` and `02-saxpy` and put the GB/s side by side.
   If they differ by more than a couple of percent, you have found a real
   effect. Use `ncu --section MemoryWorkloadAnalysis` and compare
   `dram__bytes_read` and `dram__bytes_write` against your hand count for each.
2. Write a variant that writes to a separate output `z` instead of updating `y`
   in place. Four streams now instead of three. Does the time go up by exactly
   4/3? If not, why not?
3. Compare the FLOP/s here to vector add's. SAXPY does twice the math for the
   same traffic. Did the runtime change at all? What does that confirm about
   which roof you are under?

<details>
<summary>Hints</summary>

- Does an FMA cost more than an add? How many cycles, against a 500-cycle
  memory latency?
- What fraction of the fp32 roof are you at? Could you do 100x more math per
  element for free?
- If you believe write-allocate matters here, how would you prove it with a
  counter rather than an argument?

</details>
