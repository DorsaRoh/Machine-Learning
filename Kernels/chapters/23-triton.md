# Chapter 23 - Triton

> You learned CUDA first on purpose. Triton is easier, and if you had started
> here you would have been able to write kernels without being able to predict
> them -- and when a Triton kernel is slow, the only way out is to know what it
> compiled to. Now you do.

---

## 1. What Triton actually changes

In CUDA you write code for **one thread** and reason about 32 of them at a
time. In Triton you write code for **one block**, on tensors, and the compiler
decides the thread mapping.

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=1024)
```

`offs` is a vector of `BLOCK` indices, not a scalar. `tl.load` with a mask is
the bounds check, the coalescing decision and the predication all at once.
There is no `threadIdx`, no `__syncthreads`, no shared memory declaration.

**What the compiler does for you**, which is exactly the list of things you
spent Parts I-III learning to do by hand:

| You did by hand | Triton does |
|---|---|
| thread-to-data mapping | chooses it |
| coalescing | derives it from your index expressions |
| shared memory staging | inserts it |
| `__syncthreads` placement | inserts it |
| bank-conflict swizzle | chooses the layout |
| double buffering / `cp.async` | pipelines the loop |
| `mma` fragment layouts | selects and converts |
| register tiling | allocates |

**What it does not do:** choose your algorithm, choose your tile sizes, or know
what your data looks like. Those were always the parts that mattered most, and
they are still yours.

---

## 2. The mental model that keeps you out of trouble

A Triton program instance is a CUDA threadblock. `tl.arange(0, BLOCK)` is a
block-shaped vector living in registers across the whole block. A `tl.load` of
a contiguous range is a coalesced load; a `tl.load` of a scattered range is a
gather, with every cost Chapter 22 described. `tl.dot` is `mma`.

So when you write

```python
a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
```

ask the Chapter 03 question: as the fastest-varying index moves by one, how far
does the address move? If `stride_ak` is 1, that is coalesced. If it is
`stride_am`, it is not, and Triton will not save you -- it will faithfully
compile your strided access into strided loads.

**The compiler optimizes the schedule, not the access pattern.** The access
pattern is in your index arithmetic, and it is still your job.

---

## 3. Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64,  'BN': 256, 'BK': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
```

Triton benchmarks every config on the first call for each distinct `key` value
and caches the winner. `num_warps` sets the block size; `num_stages` sets the
software pipeline depth -- the `cp.async` depth from Chapter 14, as a number
you type.

Two things to understand before you trust it:

**The search space is yours to design.** Autotuning over bad configs finds the
best bad config. Use the Chapter 12 arithmetic -- shared memory per stage,
registers per thread, blocks per SM -- to pick 6 to 12 plausible configs rather
than listing 200 and waiting.

**The first call pays for all of them.** In a benchmark loop that is fine. In
a server with varying shapes it is a latency spike on every new shape, and
`key` is what controls how often you get one.

---

## 4. Where Triton is the right answer, and where it is not

**Right answer:**

- Element-wise and fused element-wise chains. Faster to write than CUDA, and
  the generated code is usually as good.
- Reductions, softmax, LayerNorm, RMSNorm. Within a few percent of hand CUDA,
  at a tenth of the effort.
- Research: trying six variants of an attention mask before lunch.
- Anything where the win comes from fusion rather than from instruction
  scheduling. That is most things.

**Not the right answer:**

- The last 10-20% of a GEMM. CUTLASS and cuBLAS have hand-tuned epilogues,
  layouts and instruction schedules that Triton does not reach.
- Hopper-specific features. TMA, warp specialization and cluster support arrive
  in Triton later and less completely than in CUTLASS. FlashAttention-3 is
  CUTLASS for this reason.
- Anything needing explicit warp-level control: `__shfl` patterns, warp
  specialization, custom PTX.
- Anything where you must control register allocation exactly.
- Exotic data layouts the compiler has no concept of.

The useful rule: **Triton for fusion, CUDA for the last 20%.** And because you
know CUDA, you can tell which situation you are in rather than guessing.

---

## 5. Debugging Triton, which is genuinely harder

You lost the ability to print from a thread and reason about one lane. What you
get instead:

```bash
TRITON_INTERPRET=1 python my_kernel.py      # run the kernel in Python, on CPU
```

The interpreter is slow and single-threaded but you can `print` tensors and use
a debugger. It is the first thing to reach for on a correctness bug.

```python
kernel[grid](...)                           # then inspect what it produced
print(kernel.cache_key)
# TRITON_CACHE_DIR holds the generated .ttir, .ttgir, .ptx and .cubin
```

The `.ptx` and the SASS from `cuobjdump` are the ground truth, and you can read
both. When a Triton kernel is mysteriously slow, go straight there and look for
the same things Chapter 16 taught you to look for: `LDL`/`STL` (spills), the
absence of `HMMA` where you expected tensor cores, the absence of `LDGSTS`
where you expected pipelining.

`ncu` profiles Triton kernels exactly as it profiles yours, and reports the
same counters. Nothing you learned in Chapter 16 is wasted.

---

## 6. Common failure modes

**Slower than your CUDA version by 3x.** Check `num_warps` and `num_stages`
first; the defaults are frequently wrong for your shape.

**Register spills.** Your `BLOCK` sizes are too large. Triton will not warn
you; the SASS will show `LDL`/`STL`.

**A `tl.dot` that is not using tensor cores.** Check the dtypes, check that the
shapes are at least 16x16x16, and check the SASS for `HMMA`.

**Correct in the interpreter, wrong on the GPU.** Almost always a mask error
around the edges, where the interpreter's bounds happened to be benign.

**Recompiling constantly.** Something in your `key` or in your `constexpr`
arguments varies per call.

**Uncoalesced despite looking fine.** Your strides. Write out the address
expression and apply the Chapter 03 test.

---

## 7. Exercises

### 23.1 - Port the ladder

Reimplement, in Triton, and compare against your CUDA versions:

(a) `02-vector-add`. Measure both. Explain any gap.
(b) `07-reduce`. Same.
(c) `10-softmax`. Same.
(d) `10-layernorm`, fused. Same.
(e) For each, note how long the Triton version took you to write versus the
    CUDA one, and what percentage of the CUDA performance you got.
(f) Which of the four has the largest gap, and why that one?

### 23.2 - The GEMM gap

(a) Write a Triton matmul with autotuning.
(b) Measure against cuBLAS and against your `11-gemm-ladder` v5.
(c) At `M=N=K` of 512, 2048, 8192 -- does the ranking change with size?
(d) Dump the SASS. Compare the inner loop against your CUDA version's.
(e) Where exactly does the remaining gap to cuBLAS come from? Name the
    instructions.

### 23.3 - Read the generated code

For your Triton softmax:

(a) Find the `.ttgir` in the cache. What layout did it choose?
(b) Find the PTX. Where did it put shared memory, and how much?
(c) Find the SASS. How many registers per thread? Any spills?
(d) Change `num_warps` and redo all three. What moved?
(e) Predict the effect of the change on time, then measure.

### 23.4 - Autotuning honestly

(a) Autotune your Triton matmul over 20 configs. Note the winner.
(b) Using the Chapter 12 arithmetic, predict the winner from first principles.
    Did you get it right?
(c) Time the first call and the hundredth. Where did the compilation cost go?
(d) Change `M` by one element. What happens, and why?
(e) How would you deploy this in a server with 50 distinct shapes?

### 23.5 - Flash attention in Triton

(a) Port your `19-flash-fwd` to Triton.
(b) Measure against your CUDA version and against PyTorch SDPA.
(c) How many lines each? How many hours each?
(d) Which parts of the CUDA version disappeared entirely?
(e) Which parts of the Triton version are harder to reason about?
(f) Having done both: which would you choose for a new research idea, and which
    for a kernel shipping to a million users?

---

## Exercises to run

- [`23-triton-ladder`](../exercises/23-triton-ladder) - four ports, measured
  against your own CUDA. Bar: >= 80% of your CUDA version on each.

---

## What you should now believe

- Triton raises the unit of programming from the thread to the block, and
  automates the mapping, staging, synchronization and pipelining.
- It does not automate algorithm choice, tile size, or access pattern -- which
  are the decisions that set performance.
- `num_warps` and `num_stages` are block size and pipeline depth, and they are
  the first things to check when it is slow.
- Debugging is harder and the tools are the interpreter plus the generated
  PTX/SASS, which you can read.
- Triton for fusion, CUDA for the last 20%, and you can now tell which you are
  in.

Next: [Chapter 24 - Shipping](24-shipping.md), where the kernel meets a real
model and a benchmark someone will challenge.
