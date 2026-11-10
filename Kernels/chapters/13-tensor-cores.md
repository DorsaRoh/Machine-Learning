# Chapter 13 - Tensor Cores

> Everything up to here treated the FMA as the atom of computation. Tensor
> cores replace it with a small matrix multiply, raise peak throughput by
> 8-16x, and in doing so move the roofline knee so far right that *feeding*
> them becomes the entire problem.

---

## 1. What a tensor core is

A hardware unit that computes `D = A * B + C` where A, B, C, D are small
matrices, in one instruction, in a few cycles. The canonical Ampere shape is
`m16n8k16`: a 16x16 times 16x8 matrix multiply-accumulate.

```
regular FMA : 2 FLOPs per instruction per thread
m16n8k16    : 2 * 16 * 8 * 16 = 4096 FLOPs per instruction per WARP
```

Four tensor cores per SM, so an A100 gets 312 TFLOP/s in fp16 against 19.5 in
fp32. Sixteen times.

**The instruction is warp-wide.** All 32 threads cooperate; the operand
matrices are spread across the warp's registers in a specific, non-obvious
layout. You cannot call it from one thread, and you cannot choose the layout.

### The consequence for the roofline

```
A100 fp16 tensor : 312 TFLOP/s / 2.0 TB/s = 156 FLOPs per byte
H100 fp16 tensor : 990 TFLOP/s / 3.35 TB/s = 295 FLOPs per byte
```

Compare to ~10 for fp32. **Operations that were comfortably compute-bound
become memory-bound the moment you use tensor cores.** A GEMM at N=4096 has
I = 683, still fine; at N = 512, I = 85, and on an H100 that is now
memory-bound. Attention with a head dimension of 64 is around I = 64 -- firmly
memory-bound on tensor-core hardware, which is precisely why FlashAttention is
about memory and not about math.

**Every Hopper-era optimisation -- TMA, warp specialization, FP8 -- is about
feeding tensor cores fast enough.** That is the frame to hold for Chapter 14.

---

## 2. Precisions

| Type | Bits | Exp/Mantissa | Range | Notes |
|---|---|---|---|---|
| FP32 | 32 | 8/23 | ~1e38 | the baseline |
| TF32 | 19 | 8/10 | ~1e38 | fp32 range, fp16 precision. Ampere+. Drop-in for fp32 GEMM at 8x speed |
| BF16 | 16 | 8/7 | ~1e38 | fp32 range, low precision. The training default |
| FP16 | 16 | 5/10 | 65504 | more precision, overflows. Needs loss scaling |
| FP8 E4M3 | 8 | 4/3 | 448 | Hopper. Forward pass |
| FP8 E5M2 | 8 | 5/2 | 57344 | Hopper. Gradients (needs range) |

**Accumulation is always wider than the inputs.** `mma` with fp16 inputs
accumulates in fp32. This is not optional and it is why fp16 GEMM is usable at
all: the products are fp16 but the sum of 4096 of them is fp32.

**Range beats precision for training.** BF16 has 3 fewer mantissa bits than
FP16 and has essentially replaced it, because FP16's maximum of 65504 causes
overflow in attention scores and gradients, requiring loss scaling and constant
vigilance. BF16 just works. Chapter 15 goes deeper.

**TF32 is nearly free.** On Ampere+, `torch.backends.cuda.matmul.allow_tf32 =
True` gives an 8x speedup on fp32 GEMMs with a precision loss that almost never
matters. Know that it exists, and know it is on by default in some frameworks
and off in others -- which is a classic source of "why is my A100 slower than
my colleague's".

---

## 3. The WMMA API

The approachable entry point:

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void gemm_wmma(const half* A, const half* B, float* C,
                          int M, int N, int K) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + ..., lda);
        load_matrix_sync(b_frag, B + ..., ldb);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(C + ..., c_frag, ldc, mem_row_major);
}
```

A `fragment` is an opaque per-thread register bundle holding this warp's share
of a 16x16 tile. You must not index into it (its layout is unspecified and
architecture-dependent), you must have **all 32 threads of the warp** execute
every `_sync` call, and the tile dimensions are restricted to a fixed set
(16x16x16, 32x8x16, 8x32x16 for fp16).

WMMA is good for learning and for getting 60-70% of cuBLAS. Its limits are
real: you cannot control the shared-memory layout it loads from, you cannot use
`ldmatrix` explicitly, and you cannot fuse things into the fragment layout.
Every production kernel uses the next level down.

---

## 4. `mma.sync` and `ldmatrix`

PTX-level, full control:

```cuda
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
```

Now you own the register layout, and the layout is *specific*: for
`m16n8k16`, lane `i` holds particular elements of A determined by
`i/4` and `i%4`. The PTX ISA documentation has the diagrams; you will need
them, and you will refer to them every single time.

Getting data into that layout from shared memory by hand would need many
strided loads. Hence:

```cuda
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];
```

**`ldmatrix` loads four 8x8 matrices from shared memory directly into the
register layout `mma` expects**, with the shuffle done in hardware. Each of the
32 lanes supplies a row address; the hardware gathers and distributes. It is
the single instruction that makes high-performance tensor-core kernels
possible.

And it interacts with bank conflicts in a specific way, which is why
tensor-core kernels use XOR swizzles for their shared tiles rather than
padding: `ldmatrix` requires particular alignment that padding destroys.
Chapter 06 exercise 6.3, Chapter 12 section 3, and now here -- the same idea
arriving for the third time, each time more load-bearing.

---

## 5. Structuring a tensor-core GEMM

The Chapter 11 hierarchy, with the instruction tile changed:

```
threadblock tile   128 x 128 x 32     shared memory, loaded cooperatively
warp tile          64 x 64            each warp owns this
instruction tile   16 x 8 x 16        one mma.sync
```

Each warp's 64x64 tile needs `(64/16) * (64/8) = 32` `mma` instructions per
K-step of 16, with fragments held in registers across them.

The structure of the inner loop:

```
for k_block in K:
    cooperative global -> shared  (swizzled layout)
    barrier
    for k_frag in k_block step 16:
        ldmatrix A fragments for this warp
        ldmatrix B fragments
        32 x mma.sync accumulating into registers
    barrier
```

Register pressure: a 64x64 fp32 accumulator per warp is `64*64/32 = 128`
registers per thread *for accumulators alone*. You are immediately at the
limit, occupancy is ~25%, and that is correct and intended. The ILP in 32
back-to-back independent `mma` instructions is more than enough.

---

## 6. When tensor cores do not help

- **Small matrices.** The 16x16x16 granularity wastes most of the work on a
  16x16 GEMM, and launch overhead dominates anyway.
- **Memory-bound shapes.** A GEMV (`[1,K] x [K,N]`) has I = 2 and cannot use
  any of the extra FLOP/s. LLM decode is exactly this, and no amount of tensor
  core helps.
- **You cannot tolerate the precision.** Some scientific workloads genuinely
  need fp64; tensor cores do offer fp64 on A100/H100 but at a much smaller
  multiplier.
- **Odd shapes.** K not a multiple of 16 needs padding, and the padding can
  cost more than the speedup for small K.

Knowing when the fancy hardware is irrelevant is as valuable as knowing how to
use it.

---

## 7. Common failure modes

**No speedup at all.** Check the SASS for `HMMA`. If it is absent, your `wmma`
calls did not survive -- wrong `-arch`, wrong types, or the compiler could not
prove alignment.

**`misaligned address` from `load_matrix_sync`.** Fragment loads need
16-byte-aligned addresses and the leading dimension must be a multiple of 8
(fp16).

**Wrong results, structured in 16x16 blocks.** Fragment layout or the
row/col_major specification. Test with a single 16x16x16 tile and a known
matrix first.

**Hangs.** A `_sync` call inside divergent control flow. All 32 lanes, always.

**Correct but only 30% of cuBLAS.** Expected for a first WMMA kernel. The gap
is shared-memory layout, pipelining and `ldmatrix`.

**Overflow / NaN with fp16.** Scores exceeded 65504. Switch to bf16, or scale.

**fp16 accumulation gives garbage for large K.** You used `fragment<accumulator,
..., half>`. Use fp32 accumulation.

---

## 8. Exercises

### 13.1 - Re-draw your roofline

(a) Your GPU's knee for fp32, tf32, fp16-tensor, bf16-tensor (and fp8 if you
    have Hopper).
(b) Draw all of them on one plot.
(c) Mark where a 4096^3 GEMM sits. A 512^3 GEMM. A GEMV. Attention with head
    dim 64 and sequence 2048.
(d) Which of those become memory-bound when you switch from fp32 to fp16
    tensor cores?
(e) For the ones that do: what would have to change about the *algorithm* to
    make them compute-bound again? (You are now one short step from the
    FlashAttention argument.)

### 13.2 - Fragment layout

For `m16n8k16` with fp16 inputs and fp32 accumulate, using the PTX ISA
documentation:

(a) Which elements of A does lane 0 hold? Lane 1? Lane 4? Lane 31?
(b) Same for B.
(c) Same for the accumulator D.
(d) How many registers per lane for each operand?
(e) If A were stored row-major in shared memory with no swizzle, how many
    shared-memory transactions would it take to assemble lane 0's fragment
    without `ldmatrix`?
(f) How many does `ldmatrix.x4` take?

### 13.3 - Precision experiments

Compute `C = A @ B` for 4096^3 with entries drawn from N(0,1), and compare each
against a float64 reference:

(a) fp32: max relative error?
(b) tf32: same?
(c) fp16 inputs, fp32 accumulate?
(d) fp16 inputs, fp16 accumulate?
(e) bf16 inputs, fp32 accumulate?
(f) Now scale the inputs by 100. Which of the above break, and how?
(g) Rank them by error and by speed. Which is the best trade for training?
    For inference? Are those the same answer?

### 13.4 - Count the instructions

For a 128x128x32 block tile with 4 warps, fp16 inputs:

(a) `mma.sync` instructions per block per K-step.
(b) `ldmatrix` instructions.
(c) Ratio of mma to ldmatrix. Is that enough math per load?
(d) Accumulator registers per thread. Occupancy?
(e) Shared memory for the tile. Blocks per SM?
(f) Predicted TFLOP/s. Compare with cuBLAS fp16 on your GPU.

### 13.5 - When it does not help

Predict, then measure, the speedup from fp32 to fp16 tensor cores for:

(a) 4096 x 4096 x 4096
(b) 512 x 512 x 512
(c) 64 x 64 x 64
(d) 1 x 4096 x 4096 (GEMV)
(e) 4096 x 4096 x 16 (skinny K)

Two of these should show essentially no benefit. Explain each from the
roofline, not from the measurement.

---

## Exercises to run

- [`13-tensor-core-gemm`](../exercises/13-tensor-core-gemm) - WMMA to start,
  `mma.sync` for the stretch. Bar: 50% of cuBLAS fp16.

---

## What you should now believe

- A tensor core instruction is a warp-wide small matrix multiply, 8-16x the
  FLOP rate.
- The knee moves to 150-300 FLOPs/byte, so feeding them is the real problem.
- Accumulate wide; inputs can be narrow.
- BF16 beats FP16 for training because range matters more than mantissa bits.
- WMMA gets you 60-70%; `mma.sync` + `ldmatrix` + swizzle gets you the rest.
- Tensor cores do nothing for memory-bound shapes, and recognising those is
  half the skill.

Next: [Chapter 14 - Async pipelines](14-async-pipelines.md).
