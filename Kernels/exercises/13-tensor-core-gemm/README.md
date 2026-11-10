# 13 - tensor core GEMM

**Chapter:** [13](../../chapters/13-tensor-cores.md)

**Requires:** sm_70 or newer (Volta+). A T4 works.

## Task

`C = A @ B` with fp16 inputs and fp32 accumulation, 4096^3, at >= 50% of cuBLAS
fp16.

## Predict first

```
your GPU's fp16 tensor peak        = ______ TFLOP/s
your GPU's fp32 peak               = ______ TFLOP/s
ratio                              = ______x

your best fp32 kernel (from 11)    = ______ TFLOP/s
predicted WMMA kernel              = ______ TFLOP/s
predicted % of cuBLAS fp16         = ______

fp16 roofline knee for your GPU    = ______ FLOP/byte
this problem's arithmetic intensity= ______ FLOP/byte
still compute-bound?               = yes / no
```

That last pair is the point of the chapter. Check it at N = 4096 and again at
N = 512 (`make run ARGS="512 512 512"`).

## Run

```bash
make run
make run ARGS="512 512 512"       # does the knee move you off the compute roof?
make run ARGS="4096 4096 64"      # skinny K: granularity waste
```

## Pass bar

- Correctness: 2e-2 relative against cuBLAS fp16. That tolerance is loose
  because the *inputs* are fp16: the rounding happens before any multiply. If
  your error is much worse than 2e-2, you are probably accumulating in fp16 --
  check your accumulator fragment type.
- Performance: >= 50% of cuBLAS fp16 tensor-core throughput.

## Build it in this order

1. **One tile.** `M = N = K = 16`, one warp, one `mma_sync`. Verify against a
   hand-computed answer. Do not skip this; fragment bugs are invisible at
   scale and obvious at 16x16.
2. **Loop over K** with fragments loaded straight from global memory. Correct
   and slow. Measure it anyway -- you want to see how much the shared-memory
   staging buys.
3. **Stage tiles in shared memory**, load fragments from there. This is the
   Chapter 11 structure with the inner instruction swapped.
4. **Multiple warps, multiple fragments per warp.** Each warp owns a 64x64
   region = 4x8 accumulator fragments.
5. **Tune**: tile sizes, shared layout, `float4`/`uint4` staging loads.

## Verify tensor cores actually ran

```bash
make sass | grep -c HMMA
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./exercise
```

If `HMMA` count is zero, no tensor core executed and whatever you measured was
something else. This happens more often than you would think -- a wrong
`-arch`, an accumulator type the hardware cannot use, or an alignment the
compiler could not prove.

## After it passes

1. **Precision study** (Chapter 13 exercise 13.3). fp32, tf32, fp16/fp32-acc,
   fp16/fp16-acc, bf16/fp32-acc. Error against a float64 reference, and speed.
   Then scale the inputs by 100 and see which ones break.
2. **The fp16 accumulate experiment.** Switch your accumulator fragment to
   `half` and measure the error at K = 4096. Then at K = 128. Explain the
   difference. This is the most direct demonstration of why wide accumulation
   is non-negotiable.
3. **Shape sweep.** 4096, 2048, 1024, 512, 256, 128. Plot % of cuBLAS. Where
   does it fall off, and is it granularity, the roofline knee, or wave
   quantization? Distinguish them.
4. **v2: `mma.sync` + `ldmatrix`.** The stretch goal. Expect it to take a
   while. The debugging technique in `kernel.cu` -- print what each lane holds
   after an `ldmatrix` and compare to the documented layout -- is the only
   sane way in.

<details>
<summary>Hints</summary>

- A fragment is per-warp, not per-thread. How many warps does your block have,
  and how many accumulator fragments does each need for its share of the block
  tile?
- `load_matrix_sync` takes a leading dimension. For a shared tile of
  `BM x BK` halfs stored row-major, what is it? Is it a multiple of 8?
- Shared memory for fp16 tiles: 128x32 halfs = 8 KB. How many stages fit?
- If you see `HMMA` but only get 20% of cuBLAS, the problem is feeding, not
  math. Profile the stall reasons and check how much time is spent in the
  shared-memory staging.
- At `512^3`, recompute the arithmetic intensity. Is the kernel still supposed
  to be compute-bound on your GPU? If not, what is the right target, and is
  cuBLAS itself hitting the compute roof there?

</details>
