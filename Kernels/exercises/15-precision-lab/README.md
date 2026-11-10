# 15 - precision lab

**Chapter:** [15](../../chapters/15-numerics.md)

## Task

Implement four small kernels, then explain every number in the table they
produce. There is no performance gate -- the written explanations are the
deliverable.

## The four experiments

**E1 - accumulation order.** Sum 16M values, 999 out of 1000 of which are
around `1e-7` and the rest around `1.0`. Four methods: sequential in one
thread, tree, Kahan, and fp64. The spread between the first and the rest is the
whole point of Chapter 07 section 6.

**E2 - cancellation.** Row variance for `x ~ N(mu, 1)` at mu = 0, 100, 10000,
computed three ways. Watch what happens to the fp32 identity as mu grows.

**E3 - accumulator width.** fp16 dot product with fp32 vs fp16 accumulators, at
n from 128 to 1M. Find the n at which fp16 accumulation stops being usable.

**E4 - format range.** `y = x*x` computed and stored in fp32, fp16 and bf16,
for x straddling each format's limits. Note *which* values break in *which*
format, and that the two 16-bit formats break on different inputs.

## Predict first

Before running, fill in your predicted relative errors:

```
E1  sequential: ______   tree: ______   Kahan: ______   fp64: ______
E2  mu=0:    identity ______  Welford ______
    mu=100:  identity ______  Welford ______
    mu=10000:identity ______  Welford ______
E3  n=1M:    fp32 acc ______  fp16 acc ______
    the n at which fp16 accumulation loses its first digit: ______
E4  which probe values overflow in fp16? ______
    which underflow in fp16? ______
    which break in bf16? ______
```

## Run

```bash
make run
```

## Pass bar

The harness always "passes" structurally. The real bar is your write-up:

1. Explain E1's spread. Why is sequential so much worse? Why is Kahan's error
   independent of n? Is tree-fp32 better or worse than sequential-fp64, and
   what does that tell you about algorithm versus precision?
2. Derive E2's identity error from first principles and compare with the
   measurement. At mu = 10000, how many significant digits remain? Is the
   measured error consistent with that?
3. Derive E3's crossover and compare with the measurement. Why does the fp16
   accumulator fail gradually rather than suddenly?
4. For E4, give each format's max, min normal, and min subnormal, and match
   each anomalous entry in the table to one of them.

## Extensions

1. **Build a bf16-vs-fp16 adversarial pair.** An input where fp16 is correct
   and bf16 is visibly wrong, and one where the reverse holds. The second is
   easy (overflow); the first is the interesting one and requires you to think
   about mantissa bits.
2. **Measure the cost of accuracy.** Time all four E1 methods. What does Kahan
   cost on a memory-bound reduction? (The answer should be "nothing", and
   confirming that is the point.)
3. **fp64 on your GPU.** Check your chip's fp64 rate. On a consumer card it is
   1/64 of fp32. Does that change when you would reach for fp64 accumulation?
4. **Determinism audit** (Chapter 15 exercise 15.6). Add an atomic-based sum to
   E1, run it 1000 times, and report the spread in ulps.

<details>
<summary>Hints</summary>

- E1 mode 0 runs on one thread and will take a while. That is fine and
  expected; you are measuring error, not speed.
- Kahan inside a parallel reduction: where does the compensation live? Per
  thread, per warp, or per block? Does the tree combine step need compensating
  too?
- E3: the fp16 accumulator's step size near a running total of T is about
  `T * 2^-11`. At what T does an addend of ~0.3 round to nothing?
- E4: `__float2half` and `__float2bfloat16` round; what do they do on overflow?
  Check whether you get `inf` or the max finite value.

</details>
