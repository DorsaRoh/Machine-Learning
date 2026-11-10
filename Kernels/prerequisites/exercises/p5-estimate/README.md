# p5-estimate

**Chapter:** [P8 - Performance arithmetic](../../p8-performance-arithmetic.md)

> The last prerequisite, and the one that becomes Chapter 01 with no changes
> except the hardware numbers.

## Task

Count bytes and FLOPs, build the roofline model, **write down a prediction**,
then measure and find out how wrong you were. Sixty minutes.

## Run

```bash
make run
```

## Pass bar

All 28 checks pass.

## Do this in order, and do not cheat on step 2

1. Fill in part 1 (counting) and part 2 (the model). Pure arithmetic, no
   measuring.
2. **Fill in the two constants in part 3 before you write part 4.** How many
   GB/s does your machine do? How much faster will one fused pass be than
   three separate ones? Commit to numbers. The harness prints your guesses
   next to reality, and the whole value of this exercise is in that comparison.
3. Implement part 4 and run it.

A prediction written after seeing the answer is worth nothing, and the habit
this exercise is trying to build — predict, then measure, then explain the gap
— is the one that separates people who optimise kernels from people who try
things at random.

## Notes

- `copy_floats` is the **copy ceiling**: read a float, write a float, no
  arithmetic. Nothing that touches the same amount of data can beat it. The
  main course gates `02-vector-add` at 80% of exactly this number.
- The last check verifies that your model, fed your *own* measured bandwidth,
  predicts the vector-add time within a factor of three. On the reference
  machine the model says 4.65 ms and the hardware says 4.56 ms — a 2% error,
  from arithmetic done on paper. That is what a correct mental model buys you.
- Your `vector_add` may report a bandwidth slightly *above* the copy ceiling.
  That is the write-allocate effect from P8 section 4, not a miracle.

## The `sum` anomaly

Your `sum_floats` will come in at a small fraction of the copy ceiling — on
the reference machine, 5 GB/s against a 52 GB/s ceiling, ten times slower
while moving *less* data.

Work out why before reading on.

<details>
<summary>The answer</summary>

Every iteration of `acc += a[i]` must wait for the previous one to finish,
because `acc` is the input to the next add. A double add has about 4 cycles of
latency, so you get one element per 4 cycles no matter how fast memory is. The
loop is **latency-bound on a dependency chain**, not bandwidth-bound. The
copy has no such chain: every iteration is independent, so the machine runs
dozens of them at once.

The fix is to use several independent accumulators and combine them at the
end — which is to say, to turn the sum into a tree. You already wrote that
tree in `p3-floats`, for accuracy; here is the same structure appearing again
for a completely different reason.

This is why Chapter 07's reduction is a whole chapter. It is also why a GPU,
whose answer to latency is "run 2048 other threads meanwhile", is shaped the
way it is — which is the first paragraph of Chapter 00.

</details>

## After it passes

1. Add the four-accumulator version of `sum_floats` and re-measure. How close
   to the ceiling do you get? How many accumulators before it stops helping?
2. Re-run everything at `N = 100000` (small enough for L2) and at
   `N = 200000000` (far past any cache). Plot ms against N. Where are the
   knees, and what cache size does each one correspond to?
3. Change `fused` to compute something expensive — `expf` instead of a
   multiply. Does fusion still give 3x? Recompute the intensity and explain
   the new number before measuring it.
4. Look up the A100's numbers (1555 GB/s, 19.5 TFLOP/s fp32) and re-run part 2
   with them. Your laptop's ridge point is around 2 FLOP/byte; the A100's is
   12.5, and with tensor cores 200. **A faster machine is harder to feed.**
   That single sentence is why this course is 27 chapters long.
5. Now estimate, on paper: a 7B-parameter model in fp16 generating one token.
   How many bytes must be read? At 1555 GB/s, how many tokens per second is
   that, at best? Compare with a real inference server's published number. You
   have just done the calculation that the entire LLM-inference industry is
   organised around, and you will do it properly in Chapter 21.

<details>
<summary>Hints (questions only)</summary>

- `bytes_saxpy`: `y` appears on both sides of the assignment. How many times
  does it cross the memory bus?
- `bytes_dot`: the output is one scalar. Does it belong in the byte count at a
  million elements?
- `predicted_ms`: bytes divided by bytes-per-second gives seconds. What takes
  you from there to milliseconds, and which direction does the multiply go?
- `ridge_point`: the units of the answer must be FLOP/byte. Which way up does
  the division go to produce those units?
- Predicting your bandwidth: what kind of memory does your machine have, and
  what is its advertised speed? Divide by nothing; that advertised number is
  roughly the ceiling, and real code gets 60-90% of it.
- Predicting the fusion speedup: count the bytes each version moves. The ratio
  of those two counts is your prediction, and it should be an integer.

</details>
