# 10 - softmax

**Chapter:** [10](../../chapters/10-fusion-softmax-layernorm.md)

## Task

Row-wise softmax at >= 85% of copy ceiling, on **three different shapes**:

```
[8192, 1024]     the transformer case
[262144, 32]     many tiny rows
[64, 131072]     few enormous rows -- one row is 512 KB, it fits nowhere on-chip
```

The gate is on the worst of the three. One decomposition will not win all
three, and branching on the shape inside `launch_softmax` is the correct
engineering answer, not cheating.

## Predict first

```
                       decomposition   passes over data   predicted % ceiling
[8192, 1024]            ______            ______              ______
[262144, 32]            ______            ______              ______
[64, 131072]            ______            ______              ______
```

The 85% bar means one read and one write. Check your "passes over data" column
against that before writing any code -- if any row says 3, that design is
already dead.

## The stability trap

Inputs are drawn from [-10, 10]. `exp(10)` is fine, but the harness also checks
that you are not accumulating in a way that loses precision, and a future
version of you will run this on attention scores in the tens or hundreds.
Subtract the max. Always. An implementation without it will produce NaN on the
first shape and you will see it immediately.

## Run

```bash
make run        # all three shapes
make run ARGS=1 # only the first shape, while you are iterating
```

## Pass bar

- Correctness: 1e-5 relative, against a float64 CPU reference, on all shapes.
- Performance: the worst shape >= 85% of ceiling.

## The actual exercise

1. **Three-pass baseline.** Write it, measure it, confirm it lands near 33% of
   ceiling. Now you have proof that the pass count is what matters.

2. **Two-pass with caching.** Keep the row in registers or shared memory across
   passes. Measure. For `cols = 1024` and 256 threads that is 4 floats per
   thread -- what does that do to your register count and occupancy? Check with
   `-Xptxas -v`.

3. **Online softmax.** One fused pass computing max and sum together (Chapter
   10 section 2). Measure on all three shapes. On which shape does it help
   most, and why is that the shape where it matters?

4. **The huge-row case.** `cols = 131072` is 512 KB per row -- it does not fit
   in registers, shared memory, or sensibly in L2 alongside everything else.
   What is your design? How many blocks per row? How do the partial `(m, s)`
   results combine, and where does that combination happen? (You proved this
   operator associative in Chapter 10 exercise 10.3. Now you need it.)

5. **The tiny-row case.** `cols = 32`. A 256-thread block per row would waste
   87% of the block. What is the right mapping, and how many rows should one
   block handle?

6. **`__expf` vs `expf`.** Swap them and measure. How much faster, and does the
   1e-5 tolerance still pass? Look at the SASS: what does each compile to?

## After it passes

- Compare against `torch.softmax` on the same shapes. You should be
  competitive; if you are far ahead, check that PyTorch is not doing something
  extra (like a dtype conversion).
- Fuse a mask: `softmax(x + mask)` where mask is `-inf` in places. Does your
  kernel still work when an entire row is masked out? Try it. The answer
  involves a NaN and it is the exact bug that bites people in causal attention
  with padding.
- Write the backward pass: `dx = (dy - sum(dy * y)) * y`. How many passes? Can
  it be done in one, and what do you need to have saved from the forward?

<details>
<summary>Hints</summary>

- "One read and one write" does not mean one pass through your *code*. If the
  row is cached on-chip after the first read, you can traverse it three times
  in registers for free. Which shapes allow that?
- For `[262144, 32]`: how many warps does your design keep busy? Is a block per
  row leaving 7/8 of each block idle?
- For `[64, 131072]`: 64 rows on a 108-SM GPU. If you use one block per row,
  how many SMs are idle?
- Your block reduction from `07-reduce`: can it carry a `(max, sum)` pair
  instead of a float? What does the combining function look like?
- At what point does `exp` get evaluated -- once per element, or twice? Twice
  is a common accident in online softmax and it costs real time on the tiny-row
  shape.

</details>
