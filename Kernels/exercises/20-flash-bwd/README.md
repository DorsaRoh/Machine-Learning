# 20 - FlashAttention backward

**Chapter:** [20](../../chapters/20-backward-passes.md)

## Task

The fused backward: `dQ`, `dK`, `dV` without an `[N, N]` intermediate.

The harness runs the forward for you and hands you `O` and `L`, so this does
not depend on a passing `19-flash-fwd`. It does depend on understanding what
`L` is.

## Predict first

```
forward FLOPs                   : ______ GFLOP
backward FLOPs (5 matmuls)      : ______ GFLOP   ratio to forward: ______
reference HBM traffic           : ______ MB
your HBM traffic                : ______ MB
predicted speedup               : ______x
loop order you chose            : ______________________
why                             : ______________________
deterministic?                  : ______________________
```

## Run

```bash
make run
make run ARGS="1 8 2048 64 0"     # B H N d causal
make run ARGS="1 4 4096 64 0"
```

## Pass bar

- `dQ`, `dK`, `dV` each within 3e-3 of the unfused reference.
- >= 2.0x the unfused cuBLAS backward.

## Build the float64 oracle first

Chapter 20 exercise 20.5, and this is not optional advice. Before you write any
CUDA:

```python
import torch
q = torch.randn(1, 2, 16, 8, dtype=torch.float64, requires_grad=True)
# k, v likewise
o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
o.backward(torch.randn_like(o))
torch.autograd.gradcheck(lambda a, b, c:
    torch.nn.functional.scaled_dot_product_attention(a, b, c), (q, k, v))
```

Then check your derivation of `dS = P * (dP - D)` against those numbers on a
`16 x 8` instance you can print in full. An hour here saves a day later.

## Order of work

1. `compute_D`. Tiny, independent, and you can verify it alone.
2. A scalar, untiled backward: one thread per query row, loop over all keys.
   Slow, correct. This is your indexing check.
3. Add the `K`/`V` tiling and pick your loop order.
4. Shared memory, then tensor cores.

Check each gradient separately as you go. `dQ` correct tells you nothing about
`dK` -- they fail in different ways, and a transposed `dK` can look like noise
in an aggregate error metric.

## After it passes

1. **Determinism.** Run twice, compare bit-for-bit. If your `dQ` differs, you
   chose the atomics route. Implement one of the other two and measure what
   determinism cost you.
2. **Causal.** Should approach 2x. Count which `(i, j)` tile pairs you can skip.
3. **Recompute vs store.** Measure the backward with `P` stored from a forward
   (you will need the `[N, N]` buffer back) against recomputing it. Which wins,
   and at what `N` does the gap become decisive?
4. **Memory ceiling.** Find the largest `N` the reference can run, then the
   largest yours can. Quote the ratio.
5. **Compare against PyTorch.** `torch.nn.functional.scaled_dot_product_attention`
   backward on the same shape. 30% of it by hand is a good result.

<details>
<summary>Hints (questions only)</summary>

- `P_ij = exp(S_ij - L_i)` with no max pass and no row sum. Why is that valid,
  and what did the forward have to store for it to be?
- `D_i = rowsum(dO_i * O_i)`. Prove it equals `rowsum(dP_i * P_i)`. Once you
  have, you know why it can be precomputed.
- If `j` is your outer loop, what does block `j` own exclusively, and what does
  it have to share? Which of `dQ`, `dK`, `dV` is the problem?
- `dV_j += P_ij^T dO_i` -- a transpose of something that is in registers in the
  `mma` fragment layout. What are your options, and which is cheapest?
- Where does `scale` appear, and in which of the five matmuls does it not?

</details>
