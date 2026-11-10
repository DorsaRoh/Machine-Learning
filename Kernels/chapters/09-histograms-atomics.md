# Chapter 09 - Histograms and atomics

> Reduction and scan have fixed, known communication patterns. A histogram does
> not: where a thread writes depends on its data. This chapter is about
> handling *contention*, which is the third fundamental difficulty in parallel
> programming after bandwidth and latency.

---

## 1. The problem

```cuda
for (i = 0; i < n; ++i) hist[bin(x[i])]++;
```

Every thread wants to increment a counter chosen at runtime. Two threads
picking the same bin must not lose an update. That requires atomicity, and
atomicity requires serialisation, and serialisation is the enemy.

The difficulty is entirely **data-dependent**: uniformly distributed data
spreads across bins and runs fast; data where 90% of elements fall in one bin
serialises 90% of the work onto one memory location. Same kernel, same size,
20x difference. This is the first kernel you have met whose performance you
cannot predict without knowing the input distribution.

---

## 2. What an atomic actually costs

```cuda
atomicAdd(&p, v);        // read, add, write -- indivisibly
```

On modern NVIDIA hardware:

- **Global atomics execute at the L2 cache**, not in the SM and not in DRAM. An
  uncontended global atomic costs roughly an L2 access, ~200 cycles, and it is
  pipelined, so throughput can be high even though latency is not.
- **Shared-memory atomics execute at the SM**, ~30 cycles, with conflict
  resolution across the warp in hardware.
- **Contention is what costs.** k lanes of a warp hitting the same address
  serialise into k operations.

```
32 lanes, 32 different addresses : ~1 operation's worth. Fast.
32 lanes, 1 address              : 32x serialisation. Slow.
```

So the cost model is not "atomics are slow". It is:

> **An atomic costs proportionally to the number of threads contending for the
> same address at the same time.**

Every technique below is a way to reduce that number.

---

## 3. Technique 1: privatization

The single most important idea in this chapter.

Instead of every block hammering one global histogram, **give each block its
own copy in shared memory**:

```
1. zero a shared histogram                      (block-private)
2. every thread atomically increments into shared memory
3. __syncthreads()
4. one atomicAdd per bin into the global histogram
```

Contention drops from `all threads in the grid` to `threads in one block`, and
the expensive global atomics drop from `n` to `gridDim * num_bins`. For
n = 1e9 and 256 bins with 1024 blocks, that is 1e9 global atomics becoming
262,144 -- a factor of 4000.

This works whenever the output is small enough to replicate. The constraint is
shared memory: 256 bins x 4 bytes = 1 KB, trivially fine. 65,536 bins = 256 KB,
impossible. When the histogram does not fit, you partition the *bins* across
multiple passes over the data, or across blocks, and that trade (more passes
over data vs less contention) is a real design decision.

**Privatization generalises far beyond histograms.** Any time many producers
update few locations -- gradient accumulation, scatter-add, embedding backward,
MoE expert counters -- the answer is a private copy per block, then one merge.

---

## 4. Technique 2: warp aggregation

If several lanes of a warp target the same bin, combine them *before* the
atomic:

```cuda
// All lanes with the same bin elect one leader who does a single atomicAdd
// of the group's total count.
unsigned mask = __match_any_sync(0xffffffff, bin);   // sm_70+
int leader = __ffs(mask) - 1;
int count = __popc(mask);
if (lane == leader) atomicAdd(&hist[bin], count);
```

`__match_any_sync` returns, for each lane, the mask of lanes holding the same
value. One instruction, and it turns 32 atomics into one per distinct bin. For
skewed data -- precisely the case where the naive version collapses -- this is
the difference between 32x serialisation and no serialisation at all.

The compiler does a simpler version of this automatically for `atomicAdd` to a
uniform address. It cannot do the `__match_any_sync` version, because it does
not know your values are frequently equal.

### Replication within shared memory

If even a block-private histogram has hot bins, replicate it R times in shared
memory and have thread `t` use copy `t % R`, merging the R copies at the end.
Costs R times the shared memory, divides contention by R. Pick R to trade
against occupancy.

---

## 5. Technique 3: sort, then count

For extreme skew or very many bins, the answer is to stop using atomics:

```
1. compute each element's bin
2. sort by bin                     (radix sort, which is scans -- Chapter 08)
3. find segment boundaries         (a scan)
4. counts are differences of boundaries
```

No atomics, fully deterministic, and cost independent of the distribution. The
sort is expensive, so this wins only when contention is severe or bins are
numerous. It is also the structure that MoE routing uses in Chapter 22: sort
tokens by expert, then process contiguous runs.

---

## 6. Determinism

`atomicAdd` on floats is **not deterministic**: the summation order varies with
scheduling, and floating-point addition is not associative. Run the same kernel
twice, get answers differing in the last bits.

Integer atomics *are* deterministic -- integer addition is associative -- so
integer histograms are reproducible regardless.

For float accumulation where you need bit-exact reproducibility (debugging,
regression tests, reproducible training), the options are: sort-then-reduce,
fixed-order tree reduction, or fixed-point accumulation in int64. Chapter 15
goes into this properly. The important thing now is to *know* when you have
given up determinism, rather than discovering it during a debugging session six
months later.

---

## 7. The atomic family

```cuda
atomicAdd, atomicSub, atomicExch, atomicMin, atomicMax,
atomicInc, atomicDec, atomicAnd, atomicOr, atomicXor, atomicCAS
```

Types: `int`, `unsigned`, `unsigned long long`, `float` (add/exch since early
on, min/max more recently), `double` (add since sm_60), `__half2` and
`__nv_bfloat162` packed adds (sm_70/sm_80), which do two 16-bit adds in one
atomic and are important for ML gradient accumulation.

**`atomicCAS` builds anything else.** Any read-modify-write you can express as
a pure function becomes an atomic via the CAS loop:

```cuda
__device__ float atomic_max_float(float* addr, float val) {
    int* ai = (int*)addr;
    int old = *ai, assumed;
    do {
        assumed = old;
        old = atomicCAS(ai, assumed, __float_as_int(fmaxf(val,
                                                   __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
```

Know this pattern. It is how you implement a custom atomic, and the loop
structure (read, compute, CAS, retry on failure) is the same on every parallel
machine ever built.

Scope qualifiers matter for correctness in multi-GPU and host-visible memory:
`atomicAdd_block`, `atomicAdd_system`. Default is device scope.

---

## 8. Common failure modes

**Counts are too low.** You used `hist[b]++` instead of `atomicAdd`. The
non-atomic version loses updates and will still look *nearly* right on small
inputs, which is the dangerous part.

**Correct but 50x slower than expected.** Contention. Check the data
distribution before blaming the kernel.

**Fast on synthetic uniform data, slow in production.** Real data is skewed.
Always benchmark on a realistic distribution; this is the most common
benchmarking mistake in this chapter.

**Shared histogram gives wrong results.** Missing `__syncthreads()` after
zeroing it, or missing one before the merge.

**Privatized version no faster.** The bins may already be uncontended, so there
was nothing to win. Or shared-memory atomics on the hot bin are still
serialising -- try replication.

**Occupancy collapsed.** A large shared histogram, possibly replicated. Compute
`smem_per_SM / smem_per_block`.

**Float results vary run to run.** Atomics. Expected.

---

## 9. Exercises

### 9.1 - Cost the strategies

n = 1e9 elements, 256 bins, 1024 blocks of 256 threads.

(a) Global atomics only: how many atomic operations?
(b) Shared privatization: how many shared atomics, how many global?
(c) If data is uniform over 256 bins, expected lanes-per-warp colliding on one
    bin?
(d) If 90% of data falls in bin 0, same question.
(e) For (d), how many atomics does warp aggregation save?
(f) Predict the time for (a), (b) and (b)+aggregation at both distributions.
    Which pairs differ most, and why?

### 9.2 - Bin count matters

For 256, 4096, 65536 and 1e6 bins, with 48 KB of shared memory available:

(a) Does a private histogram fit?
(b) If not, what are your options? Name three and cost each.
(c) At 1e6 bins and 1e9 elements, how many elements per bin on average? What
    does that tell you about contention, and does it change your answer?
(d) At what bin count does sort-then-count become attractive? Justify with a
    rough cost model.

### 9.3 - The CAS loop

(a) Write `atomicMul` for floats using `atomicCAS`.
(b) Under contention from k threads, how many CAS attempts in the worst case?
    Expected case?
(c) Is it lock-free? Wait-free? Explain the difference and which this is.
(d) Two threads CAS the same address with the same *value*. Any problem?
    (Search: "ABA problem". Does it apply here? Why or why not?)
(e) Why does the float version go through `int` and `__float_as_int` rather
    than CAS-ing floats directly?

### 9.4 - Design before implementing

Before writing `09-histogram`, write the design down:

(a) Bins per block: private, replicated, or partitioned?
(b) Elements per thread? Coalesced reads?
(c) Warp aggregation: yes or no? At what skew does it pay for itself?
(d) The merge: one atomic per bin per block, or a separate reduction kernel?
(e) Predicted bandwidth for uniform data. For 90%-skewed data. Are they the
    same number? Should they be?

### 9.5 - Measure the skew curve

Generate data with a tunable skew (e.g. a Zipf distribution with exponent `a`,
or "fraction p goes to bin 0, the rest uniform").

(a) Sweep p from 0 to 1 and plot bandwidth for the naive global-atomic version.
(b) Same for the privatized version.
(c) Same with warp aggregation added.
(d) At which p does each strategy break down?
(e) Write the decision rule: given a measured skew, which do you ship?

This plot is the deliverable. It is also the first time in the course that the
right kernel depends on the data rather than only on the machine, and that is a
category of reasoning worth internalising.

### 9.6 - Beyond histograms

Each of these is a histogram in disguise. For each, name the hot location, the
expected contention, and the privatization strategy:

(a) Gradient accumulation for an embedding table, where many tokens in a batch
    are the same word.
(b) Counting tokens routed to each of 64 MoE experts.
(c) Scatter-add: `out[idx[i]] += x[i]` with repeated indices.
(d) Building a CSR sparse matrix from unsorted coordinates.
(e) Non-maximum suppression collecting surviving boxes.

(a) is the one that matters most in practice: an embedding backward pass on
natural language text is extremely skewed, because word frequencies are Zipf
distributed. Estimate the contention for a 50k-vocabulary table and a batch of
1M tokens.

---

## Exercises to run

- [`09-histogram`](../exercises/09-histogram) - bar: 50% of copy ceiling on
  uniform data, and you must explain the skewed case.

---

## What you should now believe

- Atomics are not slow; *contention* is slow, and it scales with the number of
  threads on one address.
- Privatization (a private copy per block, then merge) is the general answer,
  and it applies far beyond histograms.
- `__match_any_sync` plus one atomic per distinct value kills the skew case.
- When contention is extreme or bins are many, sorting removes atomics
  entirely.
- Float atomics are non-deterministic; integer atomics are not.
- Performance here depends on the data distribution, so benchmark on real data.

Next: [Chapter 10 - Fusion: softmax and LayerNorm](10-fusion-softmax-layernorm.md).
