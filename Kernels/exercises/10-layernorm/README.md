# 10 - LayerNorm

**Chapter:** [10](../../chapters/10-fusion-softmax-layernorm.md)

## Task

LayerNorm over `[8192, 4096]` fp32 at >= 85% of copy ceiling -- **and** it must
stay correct when the input is `x ~ N(1000, 1)`.

## The two gates

The harness runs your kernel twice with different input distributions:

```
x ~ N(0, 1)     well conditioned. Any formula works. This one sets the perf bar.
x ~ N(1000, 1)  mean >> std. E[x^2] - E[x]^2 subtracts two nearly-equal large
                numbers and the variance drowns in rounding error.
```

If row 1 passes and row 2 fails, your kernel is fine and your *variance
formula* is wrong. That is the designed failure, and meeting it deliberately
here is cheaper than meeting it accidentally in a training run.

## Predict first

```
compulsory traffic = ______ bytes   (x in, y out; what about gamma and beta?)
predicted time     = ______ ms
passes over x      = ______

For x ~ N(1000,1), D = 4096, fp32:
  E[x]^2   ~ ______
  E[x^2]   ~ ______
  difference = ______ , of which ______ significant digits survive
  predicted relative error in the variance : ______
```

Do that arithmetic. It is the clearest demonstration of catastrophic
cancellation you will ever construct, and you only get it by doing it.

## Run

```bash
make run
make run ARGS="8192 1024"
make run ARGS="262144 128"
```

## Pass bar

- Correctness: 1e-4 relative on **both** distributions.
- Performance: >= 85% of ceiling on the well-conditioned input.

## The actual exercise

1. **Two-pass baseline.** Mean, then variance, then normalise. Measure. How
   many times did you read `x`? What is the resulting % of ceiling?

2. **Naive one-pass.** Accumulate `sum` and `sum_sq` together, use
   `var = E[x^2] - E[x]^2`. Measure performance (should be good) and run the
   stability test (should fail). Record the actual relative error -- do not
   just note that it failed.

3. **Welford.** One pass, stable. The merge formula is in Chapter 10 section 3
   and you proved it associative in exercise 10.3. Plug it into the block
   reduction you built in `07-reduce`. Measure both gates.

4. **Cost of stability.** Welford does a division per element and a heavier
   merge. How much slower is it than the naive version on the well-conditioned
   input? If the answer is "not measurably", say why -- and then notice that
   you just got numerical safety for free, which is the actual lesson.

5. **Where do gamma and beta live?** They are `cols` floats, reused by every
   row. Global, shared, or just relying on L2? Measure all three. At
   `cols = 4096` that is 16 KB each -- what does putting them in shared memory
   do to occupancy?

6. **RMSNorm.** Drop the mean subtraction. Measure. How much faster, and why
   does it have no stability problem at all? (This is why modern LLMs use it.)

## After it passes

- **Backward pass.** `dx` requires `dgamma`, `dbeta` (reductions across rows,
  so contention -- Chapter 09) and a per-row term. Design it. Which statistics
  do you save from the forward, and which do you recompute? That trade-off is
  exactly the one FlashAttention makes in Chapter 20.
- **Fuse the residual.** `y = LayerNorm(x + residual)`. The residual is read
  anyway. Confirm the fused version costs no more than the unfused LayerNorm
  alone, and that it saves a whole kernel's traffic.
- Compare against `torch.nn.functional.layer_norm`.

<details>
<summary>Hints</summary>

- Welford's per-element update carries `(count, mean, M2)`. For a parallel
  reduction, what does one lane carry and what does the merge do with two of
  them?
- The merge formula has `n_a * n_b / n`. In a block reduction where every lane
  has processed the same number of elements, does that simplify?
- `rsqrtf` vs `1.0f / sqrtf`: which does the SASS prefer, and are they equally
  accurate? Does the 1e-4 tolerance care?
- gamma and beta are indexed by column, not by row. In a block-per-row design,
  which thread reads `gamma[c]`, and is that access coalesced?
- The second pass over `x` to compute `y`: is it a second read from HBM, or is
  `x` still in L1/registers? At `cols = 4096` the row is 16 KB. What fits?

</details>
