# 14 - pipelined GEMM

**Chapter:** [14](../../chapters/14-async-pipelines.md)

**Best on:** sm_80+ (A100, A10, RTX 30xx/40xx). On a T4 build the
register-staged double buffer instead; the gate is identical.

## Task

Paste your own best GEMM in as `gemm_baseline`, then build a multi-stage
asynchronous version that beats it by **1.15x**.

The gate is relative to *your* kernel on purpose. A weak baseline makes this
trivially easy and teaches you nothing, so do not paste your v2.

## Predict first

```
your baseline TFLOP/s              = ______
profile it: Stall Barrier          = ______%
            Stall Long Scoreboard  = ______%

time per K-step of math            = ______ cycles
memory latency                     = ______ cycles
stages needed to cover it          = ______
shared memory per stage            = ______ bytes
total shared at that depth         = ______ bytes
blocks per SM then                 = ______  (was: ______)

predicted speedup                  = ______x
```

The last two lines fight each other. More stages means better overlap and
fewer resident blocks. Predict where the optimum is before you sweep.

## Run

```bash
make run
```

Then edit `STAGES` and re-run for 2, 3, 4, 5. Plot throughput against depth.

## Pass bar

- Correctness: both versions within 1e-3 of cuBLAS.
- Performance: pipelined >= 1.15x your baseline.

## Verify the async copy actually happened

```bash
make sass | grep -c LDGSTS
```

Zero means `cp.async` did not survive compilation, and whatever speedup you
measured came from something else. Most common causes: compiled for sm_75 or
lower, copy size not 4/8/16 bytes, or alignment the compiler could not prove.

## The four bugs you will hit

In the order you will hit them:

1. **Completion is not implied.** `__syncthreads()` does not wait for
   `cp.async`. You need `__pipeline_wait_prior(n)` (or
   `cp.async.wait_group n`) *and then* a barrier.
2. **Off by one in the stage index.** You compute on stage `k % STAGES` while
   issuing into `(k + STAGES - 1) % STAGES`. Draw the ring for 4 iterations
   with STAGES=3 before coding.
3. **The tail.** The last `STAGES-1` iterations have no tile left to issue.
   Guard the issue, not the compute.
4. **Silent no-op.** Wrong size or alignment makes `cp.async` compile to
   nothing useful. Check the SASS before debugging anything else.

## After it passes

1. **Stage sweep.** Plot TFLOP/s against STAGES, with occupancy on a second
   axis. Mark the predicted optimum. Were you right?
2. **Which stall did you fix?** Profile before and after and compare
   `Stall Barrier` and `Stall Long Scoreboard`. If neither moved but the time
   dropped, find what did move.
3. **`.cg` vs `.ca`.** Use the cache-global variant so streaming tile data does
   not pollute L1. Measure the difference.
4. **`cp.async.ignore-src`** for out-of-bounds tiles: removes the boundary
   branches entirely. Try it on a non-multiple size like 4000.
5. **Apply it to attention.** This same pipeline structure is what
   `19-flash-fwd` wants. You are building the tooling for Part V here.
6. **If you have Hopper:** read the TMA path in the CUTLASS SM90 kernels and
   identify what a descriptor-driven copy removes from your inner loop. Count
   the address-arithmetic instructions your version issues per tile.

<details>
<summary>Hints</summary>

- Draw the ring buffer state for STAGES=3 across 5 iterations: which stage is
  being filled, which is being read, which is idle at each step.
- `__pipeline_wait_prior(N)` waits until at most N commit-groups are
  outstanding. If you have STAGES-1 groups in flight and want the oldest, what
  is N?
- Why must the barrier come *after* the wait rather than before?
- Your baseline staged through registers. Did you delete that path, or are you
  now doing both?
- If the pipelined version is slower: check blocks per SM. Shared memory just
  went up by a factor of STAGES/2.

</details>
