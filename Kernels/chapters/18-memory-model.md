# Chapter 18 - The memory model and races

> The bugs in this chapter are the ones that pass your tests, ship, and then
> fail once a week in production on a different GPU. They are also the ones
> that separate people who write kernels from people who can be trusted with
> them.

---

## 1. What a race is

A data race: two threads access the same location, at least one writes, and
nothing orders them. The result is undefined -- not "one of the two values",
*undefined*, because the compiler is allowed to assume races do not happen and
optimise accordingly.

```cuda
// RACE: thread i writes s[i], thread i+1 reads it. Nothing orders them.
__shared__ float s[256];
s[tid] = x[tid];
float v = s[tid + 1];      // may read the old value, the new one, or garbage
```

The fix is a barrier, and the discipline is: **between a write and a read of
the same shared location by different threads, there must be a
synchronisation.**

```cuda
s[tid] = x[tid];
__syncthreads();
float v = s[tid + 1];
```

---

## 2. The barriers

### `__syncthreads()`

All threads in the block. Two guarantees: execution does not proceed until
everyone arrives, and all shared and global writes made before it are visible
to everyone after it.

**It must be reached by every thread in the block.** This is undefined
behaviour:

```cuda
if (tid < 100) {
    __syncthreads();   // threads 100+ never arrive
}
```

On real hardware it hangs, or completes with corrupted state, or works fine on
one GPU and fails on another. `compute-sanitizer --tool synccheck` finds it.

The two-barrier tiling pattern from Chapter 06, restated because it is where
this bites in practice:

```cuda
for (tile : tiles) {
    load_into_shared();
    __syncthreads();      // (1) data is ready
    compute_from_shared();
    __syncthreads();      // (2) everyone is done before the next load overwrites
}
```

Omitting (2) is the single most common CUDA race. It usually works, because
blocks tend to stay roughly in step, and it fails under load or on a different
architecture.

### `__syncwarp(mask)`

Synchronises the named lanes of one warp. Required on Volta+ wherever you
previously relied on implicit lockstep (Chapter 05).

### Cooperative groups

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

auto block = cg::this_thread_block();
block.sync();                                  // like __syncthreads
auto warp = cg::tiled_partition<32>(block);
warp.sync();
float total = cg::reduce(warp, v, cg::plus<float>());

auto grid = cg::this_grid();
grid.sync();   // GRID-WIDE, requires cudaLaunchCooperativeKernel
```

`grid.sync()` is the one new capability: a barrier across all blocks. It
requires a cooperative launch and a grid small enough that all blocks are
*simultaneously resident* -- otherwise it deadlocks, because a block that has
not been scheduled can never arrive. Use
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` to size the grid, always.

It is the right tool for iterative algorithms (graph traversal, multigrid,
anything with global phases) where relaunching a kernel per iteration costs
more than the constrained grid size.

---

## 3. Fences versus barriers

**A barrier makes threads wait. A fence orders memory operations.** They are
not the same and the distinction matters.

```cuda
__threadfence_block();   // order my accesses as seen by my block
__threadfence();         // ...as seen by the whole device
__threadfence_system();  // ...as seen by the device AND the host
```

A fence does not block anyone. It guarantees that every memory operation issued
before it becomes visible before any issued after it. The canonical use is
publishing a value with a flag:

```cuda
// Producer
data[i] = value;
__threadfence();          // without this, the flag may be visible first
flag = 1;

// Consumer
while (atomicAdd(&flag, 0) == 0) { }   // spin
float v = data[i];                      // now guaranteed to see `value`
```

Without the fence, the hardware or compiler may make `flag = 1` visible before
`data[i] = value`, and the consumer reads garbage. This is exactly the bug that
makes a decoupled look-back scan (Chapter 08) fail once in ten thousand runs,
and `racecheck` will not find it because it is a global-memory ordering issue,
not a shared-memory race.

---

## 4. volatile, atomic, and the compiler

```cuda
volatile int* flag;   // forces a real load/store every time
```

`volatile` stops the compiler caching the value in a register -- which is
necessary for a spin loop to ever terminate -- but it provides **no ordering
guarantees at all**. It is not a substitute for a fence, and code that uses
`volatile` for synchronisation is relying on luck.

The modern, correct tool is atomics with explicit scope and ordering:

```cuda
#include <cuda/atomic>
cuda::atomic<int, cuda::thread_scope_device> flag;
flag.store(1, cuda::memory_order_release);
while (flag.load(cuda::memory_order_acquire) == 0) { }
```

Acquire-release gives you the ordering the fence was providing, attached to the
operation that needs it, checkable by the compiler. If you are writing
inter-block synchronisation in new code, use this rather than
`volatile` + `__threadfence`.

---

## 5. compute-sanitizer

```bash
compute-sanitizer --tool memcheck   ./exercise   # out of bounds, misaligned
compute-sanitizer --tool racecheck  ./exercise   # shared memory races
compute-sanitizer --tool initcheck  ./exercise   # uninitialised global reads
compute-sanitizer --tool synccheck  ./exercise   # illegal barrier usage
```

Run `memcheck` and `racecheck` on every kernel before you believe any
performance number. They are slow (10-100x) so you run them on small inputs,
and they catch classes of bug that no amount of testing will.

**What racecheck cannot find:**

- Global memory races between blocks (it only analyses shared memory).
- Ordering bugs where both accesses are atomic but the *sequence* is wrong.
- Races that need a specific scheduling your run did not produce.

So it is necessary, not sufficient. Inter-block protocols need reasoning.

---

## 6. Writing correct inter-block protocols

If you find yourself synchronising between blocks, work through this list:

1. **Can you avoid it?** Two kernel launches give you a free, correct global
   barrier. The cost is a launch (5 us) plus re-reading data. Often that is
   cheaper than the engineering risk.
2. **Is the grid guaranteed resident?** If a block spins waiting for a block
   that has not been scheduled, you deadlock. Blocks are *not* guaranteed to
   run concurrently or in `blockIdx` order.
3. **Assign work by atomic counter, not by `blockIdx`.** This makes the
   dependency order match the scheduling order, which is what makes decoupled
   look-back safe.
4. **Order your publishes.** Value, then fence (or release store), then flag.
5. **Use 64-bit packed atomics** where you can, so value and status are updated
   together and the ordering question disappears.
6. **Test adversarially.** Run 10,000 times. Vary the grid size. Run on a
   different GPU. Run with other work on the GPU to perturb scheduling.

---

## 7. Common failure modes

**Hangs on some runs.** Divergent `__syncthreads`, or a spin loop waiting on an
unscheduled block, or `grid.sync()` with an oversized grid.

**Intermittently wrong, more often with more blocks.** Global race or missing
fence.

**Correct in debug (`-G`), wrong in release.** A race the optimiser exposed, or
`volatile` doing less than you assumed. Debug builds serialise enough to hide
it.

**Correct on an A100, wrong on an H100.** Warp-synchronous assumptions, or
timing-dependent races. Newer hardware reorders more aggressively.

**Spin loop never exits.** The value is cached in a register. You need
`volatile` or an atomic load.

**racecheck is clean and it is still wrong.** It is a global-memory or ordering
bug. Reason about it; the tool cannot help.

**First run correct, subsequent runs wrong.** Uninitialised state that happened
to be zero the first time. `initcheck`.

---

## 8. Exercises

### 18.1 - Find the race

Each snippet has a bug. Name it, say when it manifests, and say which sanitizer
tool (if any) finds it.

```cuda
// (a)
__shared__ float s[256];
s[tid] = in[gid];
out[gid] = s[255 - tid];

// (b)
for (int t = 0; t < tiles; ++t) {
    s[tid] = in[t * 256 + tid];
    __syncthreads();
    acc += s[tid] * s[255 - tid];
}

// (c)
if (tid == 0) s_flag = 0;
__syncthreads();
if (cond(tid)) atomicAdd(&s_flag, 1);
if (s_flag > 0) do_something();

// (d)
data[blockIdx.x] = compute();
flag[blockIdx.x] = 1;
if (blockIdx.x > 0) {
    while (flag[blockIdx.x - 1] == 0) { }
    acc = data[blockIdx.x - 1];
}

// (e)
__shared__ float s[32];
if (tid < 32) {
    s[tid] = v;
    s[tid] += s[(tid + 16) % 32];
}
```

(d) has at least three separate problems. Find all of them.

### 18.2 - Build a grid barrier

(a) Implement a grid-wide barrier with an atomic counter and a generation
    (phase) flag.
(b) Prove it cannot deadlock, *given* that all blocks are resident.
(c) How do you guarantee that precondition? Which API tells you the maximum
    safe grid size?
(d) Compare its cost against two kernel launches, at grid sizes of 100 and
    10,000 blocks.
(e) At what point does relaunching win?

### 18.3 - Make something fail

Take your `06-transpose` v3 and delete the second `__syncthreads()`.

(a) Does it still give the right answer? At n = 1024? At n = 16384?
(b) Does `racecheck` find it?
(c) Run it 1000 times. How many failures?
(d) Change the block size. Does the failure rate change?
(e) Write one paragraph on why "it worked when I tested it" is not evidence of
    correctness in parallel code.

### 18.4 - The publish-subscribe protocol

Implement a two-block protocol: block 0 computes a value and publishes it;
block 1 waits and consumes it.

(a) Naive version with a plain flag. Does it work? Always?
(b) Add `volatile`. Better? Is it correct?
(c) Add `__threadfence()`. Now correct?
(d) Rewrite with `cuda::atomic` acquire/release. Compare the generated SASS
    with (c).
(e) Pack value and flag into one 64-bit atomic. Does that remove the need for
    the fence? Why?

### 18.5 - Audit everything you have written

Run `memcheck` and `racecheck` on every exercise from Parts I-III.

(a) Which ones report problems?
(b) For each report, is it a real bug or a false positive? (racecheck has
    false positives around warp-synchronous code it cannot prove safe.)
(c) Fix the real ones.
(d) Which bugs were present but not causing visible failures? Those are the
    ones worth thinking hardest about.

---

## Exercises to run

- [`18-race-hunt`](../exercises/18-race-hunt) - find and fix four planted bugs.
  Bar: sanitizer clean and bit-reproducible over 500 runs.

---

## What you should now believe

- A race is undefined behaviour, not "one of two values".
- `__syncthreads()` must be reached by every thread in the block.
- Barriers make threads wait; fences order memory. You often need both.
- `volatile` prevents caching and provides no ordering. Use atomics with
  acquire/release.
- `racecheck` covers shared memory only; inter-block correctness needs
  reasoning.
- Blocks are not guaranteed to be co-resident or ordered, so spin-waiting on
  another block can deadlock.
- "It worked when I tested it" is not evidence.

Next: [Chapter 19 - FlashAttention](19-flash-attention.md). Part IV is done;
you can now write kernels that are fast *and* trustworthy.
