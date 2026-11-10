# Chapter 08 - Scan (prefix sum)

> A reduction produces one number. A scan produces `n` numbers, each depending
> on everything before it. That sequential-looking dependency is why scan feels
> impossible to parallelise, and why the algorithms that do it are the most
> elegant things in this subject.

---

## 1. The problem

```
inclusive scan:  y[i] = x[0] + x[1] + ... + x[i]
exclusive scan:  y[i] = x[0] + x[1] + ... + x[i-1],   y[0] = identity
```

```
x          : [3, 1, 7, 0, 4, 1, 6, 3]
inclusive  : [3, 4, 11, 11, 15, 16, 22, 25]
exclusive  : [0, 3, 4, 11, 11, 15, 16, 22]
```

Exclusive is the one you usually want, because `y[i]` is "where my output
starts" -- the allocation offset for element `i`. That is what makes scan the
engine of:

- **Stream compaction**: keep elements matching a predicate. Scan the 0/1 flags
  to get output positions.
- **Sorting**: radix sort is a scan of digit counts per pass.
- **Sparse formats**: CSR row pointers are an exclusive scan of row counts.
- **Variable-length batching**: offsets into a ragged tensor are a scan of
  lengths. This is how every modern LLM serving stack packs sequences.
- **Allocation**: any "each thread needs a variable number of slots" problem.

Scan is less famous than reduction and far more useful.

---

## 2. Why it is not obviously parallel

`y[i]` depends on `y[i-1]`. Naively that is a chain of length `n` -- completely
serial. The insight that breaks it is **associativity**: you may regroup.

```
y[7] = ((x0+x1) + (x2+x3)) + ((x4+x5) + (x6+x7))
```

The partial sums computed on the way up a reduction tree are exactly the
building blocks every prefix needs. Both algorithms below exploit this; they
differ in how they arrange the bookkeeping.

---

## 3. Hillis-Steele (inclusive, step-efficient)

```
for d = 1, 2, 4, 8, ... < n:
    for all i in parallel:
        if i >= d: x[i] += x[i - d]
```

```
init   : [3,  1,  7,  0,  4,  1,  6,  3]
d=1    : [3,  4,  8,  7,  4,  5,  7,  9]
d=2    : [3,  4, 11, 11, 12, 12, 11, 14]
d=4    : [3,  4, 11, 11, 15, 16, 22, 25]
```

- **Steps:** `log n`. Shortest possible depth.
- **Work:** `n log n` additions.
- **Problem:** the read of `x[i-d]` and the write to `x[i]` must not interleave.
  With everything in one array you need a barrier *and* double buffering, or
  you read a value someone already updated.

Work-inefficient by a factor of `log n`, but with the shortest critical path.
That trade -- more total work, less depth -- is the recurring theme of parallel
algorithms.

**Within a warp it is nearly free**, because `__shfl_up_sync` does the shifted
read with no barrier and no memory:

```cuda
__device__ float warp_scan_inclusive(float v) {
    for (int d = 1; d < 32; d <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, d);
        if (threadIdx.x % 32 >= d) v += n;
    }
    return v;
}
```

Five instructions-ish for a 32-element scan. This is the primitive everything
else is built on.

---

## 4. Blelloch (exclusive, work-efficient)

Two phases over a balanced tree.

**Up-sweep** (reduce): build partial sums bottom-up, leaving the total at the
root.

```
init    : [3,  1,  7,  0,  4,  1,  6,  3]
d=1     : [3,  4,  7,  7,  4,  5,  6,  9]
d=2     : [3,  4,  7, 11,  4,  5,  6, 14]
d=4     : [3,  4,  7, 11,  4,  5,  6, 25]   <- 25 is the total
```

**Down-sweep**: set the root to the identity, then at each level the left child
takes the parent's value and the right child takes parent + old left.

```
set root 0 : [3,  4,  7,  0,  4,  5,  6, 25]  (save 25 if you want the total)
d=4        : [3,  4,  7,  0,  4,  5,  6, 11]
d=2        : [3,  0,  7,  4,  4, 11,  6, 16]
d=1        : [0,  3,  4, 11, 11, 15, 16, 22]  <- exclusive scan
```

- **Steps:** `2 log n`.
- **Work:** `2n` additions. Asymptotically optimal.
- **Problem:** half the threads idle at each level, and the strided indexing
  causes severe bank conflicts unless you add a conflict-avoiding offset
  (`CONFLICT_FREE_OFFSET(i) = i >> 5` added to every shared index).

In practice, on a GPU, the step-efficient version often wins at block scale
despite doing more work, because the extra additions are free (you are memory-
bound) while the extra steps and barriers are not. **Work efficiency is a CPU
intuition that misleads on GPUs.** Verify this yourself rather than taking it
on faith -- it is exercise 8.4.

---

## 5. Scanning an array larger than a block

The classic three-kernel structure:

```
1. each block scans its own chunk, writes its chunk total to block_sums[]
2. scan block_sums[]                        (recursively, or one block)
3. each block adds block_sums_scanned[blockIdx] to its chunk
```

Correct and simple. Cost: **the data is read and written three times**, so a
memory-bound kernel runs at a third of its potential. For n = 256M that is 3x
too slow, and the structure cannot be fixed by tuning.

### Decoupled look-back: a single pass

The modern answer (Merrill & Garland; it is what CUB uses) scans in **one pass
over the data**, by having each block publish its state and look backward at
its predecessors:

```
each block, in order of a dynamically assigned tile index:
  1. compute its local sum
  2. publish it with status AGGREGATE        (visible to successors)
  3. look backward at predecessor blocks:
       - if a predecessor has status PREFIX (inclusive prefix known),
         add it and stop -- you now know your own prefix
       - if it has status AGGREGATE, add its aggregate and keep looking back
  4. publish its own inclusive prefix with status PREFIX
  5. scan its tile locally and add the prefix
```

Traffic: one read, one write. **Optimal.** The price is a genuinely subtle
piece of concurrent programming: you need the right memory ordering
(`__threadfence`, volatile or atomic loads) so that a successor cannot observe
a status flag before the value it describes. Get the fence wrong and you get a
rare, data-dependent wrong answer that no unit test catches.

It also relies on blocks making progress in roughly launch order, which is why
the tile index is assigned by an atomic counter rather than taken from
`blockIdx` -- that makes the dependency order match the *scheduling* order,
without which a block could spin forever waiting on a predecessor that has not
been scheduled.

You are not required to implement look-back to pass `08-scan`. You are required
to understand why three passes cost 3x, and to be able to explain the look-back
protocol. Implementing it is the stretch goal, and it is a genuinely hard,
genuinely worthwhile piece of work.

---

## 6. Segmented scan

Scan within segments, resetting at boundaries. Used for ragged batches, CSR
sparse matrices, and per-sequence operations in packed transformer inputs.

```
values : [3, 1, 7 | 0, 4 | 1, 6, 3]
scan   : [3, 4, 11| 0, 4 | 1, 7, 10]
```

The trick: define a combining operator on `(value, flag)` pairs that resets
when it crosses a segment head, and verify it is associative. Then the same
machinery works unchanged. Finding that operator is exercise 8.6, and the
technique -- build an associative operator over an augmented type -- is the
same move that makes online softmax work.

---

## 7. Common failure modes

**Off by one everywhere.** Inclusive vs exclusive. Write the definition down
before coding.

**Right for n = 256, wrong for n = 1000.** Non-power-of-two. Pad to a power of
two with the identity, or handle the boundary explicitly.

**Intermittently wrong, worse with more threads.** The Hillis-Steele
read-then-write hazard. You need two barriers per step, or two buffers.

**Blelloch is slower than Hillis-Steele.** Expected on a GPU, but check bank
conflicts before accepting it -- the strided tree indexing is conflict-heavy
without an offset.

**Three-kernel version stuck at ~1/3 of bandwidth.** That is the algorithm, not
a bug. The fix is a different algorithm.

**Look-back version hangs.** A block is waiting on a predecessor that has not
been scheduled. Use an atomically assigned tile index, not `blockIdx`.

**Look-back version wrong once in 10,000 runs.** Memory ordering. The status
flag became visible before the value it refers to. This is Chapter 18 and it is
why that chapter exists.

---

## 8. Exercises

### 8.1 - By hand

`x = [2, 5, 1, 3, 8, 0, 4, 7]`.

(a) Inclusive scan.
(b) Exclusive scan.
(c) Trace Hillis-Steele, writing the array after each of the 3 steps.
(d) Trace Blelloch up-sweep and down-sweep, all 6 steps.
(e) Count additions in each. Which does fewer? Which has fewer steps?
(f) Count barriers in each, for a block-level implementation.

### 8.2 - The hazard

In Hillis-Steele, `x[i] += x[i-d]` for all `i` at once.

(a) Construct a concrete 4-element case where one barrier per step gives the
    wrong answer.
(b) Where exactly must barriers go to make it correct?
(c) How does double buffering remove one of them?
(d) Within a warp, `__shfl_up_sync` needs neither. Why not?

### 8.3 - Warp scan to block scan

(a) Write a 32-element inclusive scan using `__shfl_up_sync`. Trace it for
    `v = 1` in every lane.
(b) Build a 1024-element block scan on top: each warp scans itself, warp totals
    are scanned, then each warp adds its offset. How many barriers?
(c) How much shared memory?
(d) Compare with a pure shared-memory Hillis-Steele block scan: barrier count,
    shared traffic, instruction count. Which do you expect to win and by how
    much?

### 8.4 - Work efficiency is a lie (test it)

(a) Predict the ratio of total additions, Hillis-Steele vs Blelloch, at
    block size 1024.
(b) Predict which is faster on your GPU, and by how much.
(c) Implement both in `08-scan` and measure.
(d) If the work-inefficient one wins, explain precisely what resource the extra
    additions were free in.
(e) At what hypothetical ratio of compute-to-bandwidth would Blelloch start to
    win? Is any real GPU near that ratio?

### 8.5 - Count the passes

(a) For the three-kernel scan at n = 256M fp32, count every byte read and
    written across all three kernels.
(b) Divide by your measured ceiling: predicted time.
(c) Same for a one-pass look-back scan.
(d) Ratio. Does it match the 3x you expect, or is it less? What else does the
    three-kernel version pay that the byte count does not capture?

### 8.6 - Segmented scan operator

Define a combining operator on pairs `(value, is_segment_head)` that computes a
segmented inclusive scan.

(a) Write the operator.
(b) Prove it is associative. (This is a real proof; do it by cases.)
(c) What is the identity element?
(d) Can `__shfl_up_sync` carry this type? What changes?
(e) Use it to compute, in one pass, the row offsets of a CSR matrix from a
    sorted list of row indices.

### 8.7 - Stream compaction

Combine Chapter 05's ballot trick with scan: copy the elements of `x`
satisfying a predicate into a dense output, order-preserving, in one pass.

(a) Within a warp, use ballot + popcount. Within a block?
(b) Across blocks?
(c) Count the passes over the data. Can it be done in one?
(d) At a 50% keep rate, predict the bandwidth relative to a plain copy.
(e) Does the keep rate change the answer? Try 1% and 99% and explain the shape.

---

## Exercises to run

- [`08-scan`](../exercises/08-scan) - exclusive scan, bar 60% of copy ceiling
  (three-kernel structures cannot pass; plan accordingly).

---

## What you should now believe

- Scan is a reduction whose intermediate results are all kept, and it is the
  foundation of compaction, sorting, and all ragged data structures.
- Hillis-Steele: `log n` steps, `n log n` work. Blelloch: `2 log n` steps, `2n`
  work. On a GPU the step-efficient one usually wins.
- Shuffles make warp-level scan nearly free; block scan is warp scan twice.
- Three-kernel scan costs 3x the traffic, and decoupled look-back is how you
  get it back.
- Augmenting the element type to make an operator associative is a general and
  very powerful move.

Next: [Chapter 09 - Histograms and atomics](09-histograms-atomics.md).
