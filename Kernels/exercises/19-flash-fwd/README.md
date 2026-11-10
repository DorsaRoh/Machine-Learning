# 19 - FlashAttention forward

**Chapter:** [19](../../chapters/19-flash-attention.md)

## Task

Fused attention forward. No `[N, N]` intermediate, anywhere, ever.

The harness gives you a workspace of `4 * B * H * N` floats. That is not an
oversight: it is `O(N)`, and an implementation that wants to materialize `S`
will not fit in it. If you find yourself needing more, stop and re-read
chapter 19 section 4.

## Predict first

Do chapter 19 exercises 19.1, 19.2 and 19.3 before you write code. Specifically,
fill this in for your chosen `BR`, `BC`:

```
registers/thread for O          : ______   (of 255)
registers/thread, everything    : ______
shared memory per block         : ______ bytes
blocks per SM                   : ______
HBM traffic, baseline           : ______ MB
HBM traffic, yours              : ______ MB
predicted speedup from traffic  : ______x
non-matmul fraction of the loop : ______%
```

If your predicted speedup is below 2.5x, your tile sizes are wrong and no
amount of coding will fix it.

## Run

```bash
make run
make run ARGS="1 8 2048 64 0"     # B H N d causal
make run ARGS="1 8 4096 64 0"
make run ARGS="1 8 2048 64 1"     # causal
```

## Pass bar

- `O` within 2e-3 of the three-kernel baseline.
- `L` (the per-row logsumexp, `m + log(l)`) within 2e-3. Exercise 20 needs it,
  so a kernel that skips it is not finished.
- >= 2.5x the three-kernel cuBLAS baseline at the default shape.

## Order of work

Resist writing the tensor-core version first. The sequence that works:

1. **Plain fp32, one thread per query row, no tiling.** Slow, obviously
   correct, and it establishes that your indexing and your `L` are right.
2. **Tile over `K`/`V` with the online softmax recurrence.** Still scalar math.
   Confirm correctness is unchanged. This is the step where the algorithm is
   actually implemented; everything after is performance.
3. **Shared memory for `K_j`, `V_j`.** Measure.
4. **`wmma` or `mma.sync` for both matmuls.** Chapter 13. This is where the
   fragment layout between the two matmuls becomes the problem.
5. **Tune `BR`, `BC`, warps per block.** Re-predict before each change.

Keep step 2 around under an `#if` so you can bisect when step 4 breaks.

## Debugging it

- Run at `N = 128` first, where you can print the whole thing.
- Check one row at a time. A per-row factor that drifts is the rescale; a
  constant factor is `scale`; garbage in the last partial tile is the bounds
  guard.
- If `O` is right but `L` is wrong, you are normalizing somewhere you should
  not be.
- `compute-sanitizer --tool memcheck` before you trust any timing.

## After it passes

1. **Causal.** `ARGS="1 8 2048 64 1"` should be close to 2x faster than
   non-causal, not 1.1x. If it is 1.1x, you are masking rather than skipping.
2. **Sweep `N`** from 512 to 16384. Plot your time and the baseline's. The
   baseline will OOM; note at which `N`, and note that your kernel does not.
3. **Sweep `d`**: 32, 64, 128. At which `d` does your register budget break,
   and what did you change?
4. **Profile it.** `dram__bytes.sum` against your section-5 prediction. If it
   is 3x your prediction, find the extra traffic before tuning anything else.
5. **Compare against PyTorch.** In Colab, time
   `torch.nn.functional.scaled_dot_product_attention` on the same shape. What
   fraction of it do you reach? Anything above 40% by hand is respectable;
   below 20% means a specific thing is wrong and the profiler knows what.
6. **Multi-query / grouped-query attention**: `K` and `V` shared across heads.
   What changes in the kernel, and what changes on the roofline?

<details>
<summary>Hints (questions only)</summary>

- `O` must be rescaled before `P_j @ V_j` is added. Why not after?
- Where does `1/l` get applied, and how many times? What is the cost of getting
  that wrong by one iteration?
- `rowmax(S_j)`: in your layout, which threads hold row `r` of `S_j`? Is that a
  shuffle, a shared-memory reduction, or neither?
- `exp` or `exp2`? What has to change elsewhere if you switch, and how much
  does it buy? (Count it before you do it.)
- The second matmul's A operand is the first matmul's output. Are the fragment
  layouts the same? If not, what is the cheapest way to bridge them?
- For causal: which `j` tiles can you skip entirely, and which need an
  element-wise mask?

</details>
