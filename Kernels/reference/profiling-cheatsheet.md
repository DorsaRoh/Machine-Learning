# Profiling cheat sheet

Which tool answers which question, and which counter settles which argument.

## Start here: what kind of problem is it?

```
Is the GPU busy at all?                 -> nsys    (timeline, gaps, copies)
   |-- no: launch overhead, CPU bound, serialisation, missing streams
   |-- yes
        |
        Is this kernel memory- or compute-bound?   -> ncu "Speed Of Light"
           |-- memory high, compute low  -> move fewer bytes (coalescing, fusion, tiling)
           |-- compute high, memory low  -> fewer/cheaper instructions, tensor cores
           |-- BOTH low                  -> latency-bound: not enough work in flight
```

That last case is the one beginners misread. Low on both axes never means
"balanced". It means the machine is waiting.

## Nsight Compute, in order of usefulness

```bash
ncu --set full ./exercise                       # everything, slow
ncu --section SpeedOfLight ./exercise           # the first question
ncu --section MemoryWorkloadAnalysis ./exercise # the memory chart
ncu --section WarpStateStats ./exercise         # why warps are not issuing
ncu --section Occupancy ./exercise              # achieved vs theoretical
ncu --kernel-name my_kernel --launch-skip 5 --launch-count 1 ./exercise
ncu --set full -o report ./exercise             # write report.ncu-rep for the GUI
ncu --import report.ncu-rep --page details      # read it back in the terminal
```

Profile a *warmed-up* launch (`--launch-skip`). The first launch includes
one-time costs and will slander your kernel.

## The counters that matter

| Question | Metric | Read it as |
|---|---|---|
| Am I coalesced? | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `...op_ld.sum` requests | 4 sectors/request is perfect for fp32; 32 is worst case |
| How many bytes actually hit HBM? | `dram__bytes_read.sum`, `dram__bytes_write.sum` | compare to the bytes your algorithm *needs*; the ratio is your waste |
| What fraction of peak bandwidth? | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | >80% means stop optimising memory |
| What fraction of peak compute? | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | |
| Tensor cores actually used? | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | 0 means your `wmma` did not survive compilation |
| Shared memory bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | should be ~0; anything large is a layout bug |
| Am I actually resident? | `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy, vs theoretical from the occupancy calculator |
| Did I spill? | `ptxas -v` at compile time, or `local_load`/`local_store` counters | any spill in an inner loop is a red alert |
| Branch divergence | `smsp__thread_inst_executed_per_inst_executed.ratio` | 32 = no divergence, lower = lanes idling |

## Stall reasons, decoded

`WarpStateStats` tells you what resident-but-not-issuing warps are waiting for.
This is the single most diagnostic section in the profiler.

| Stall | Means | Usual fix |
|---|---|---|
| `Stall Long Scoreboard` | waiting on a global memory load | more ILP, prefetch, `cp.async`, better reuse |
| `Stall Short Scoreboard` | waiting on shared memory / MIO | bank conflicts, or too much shared traffic |
| `Stall Barrier` | waiting at `__syncthreads()` | load imbalance in the block; fewer barriers; double buffer |
| `Stall MIO Throttle` | memory I/O queue full | too many shared/global instructions; vectorise |
| `Stall LG Throttle` | local/global queue full | too many outstanding requests; often spilling |
| `Stall Wait` | fixed-latency dependency (e.g. FMA chain) | break dependency chains, more independent accumulators |
| `Stall Not Selected` | ready, but another warp was picked | healthy -- this is the scheduler having choices |
| `Stall Math Pipe Throttle` | the math pipe is saturated | genuinely compute-bound; use a wider pipe |
| `Stall IMC/Drain/Dispatch` | instruction cache, exit, dispatch | giant unrolled loops; too many tiny kernels |

`Stall Not Selected` dominating is good news. `Stall Long Scoreboard`
dominating means you are memory-latency-bound and occupancy or prefetching is
the lever.

## Nsight Systems

```bash
nsys profile -o timeline --trace=cuda,nvtx,osrt ./exercise
nsys stats timeline.nsys-rep
```

Use it to answer: are my kernels overlapping? is the GPU idle between them? is
a `cudaMemcpy` blocking? is the CPU failing to feed the GPU? Annotate your own
phases with NVTX ranges (`nvtxRangePush/Pop`) to make the timeline readable.

## compute-sanitizer

```bash
compute-sanitizer --tool memcheck   ./exercise   # out-of-bounds, misaligned
compute-sanitizer --tool racecheck  ./exercise   # shared memory races
compute-sanitizer --tool initcheck  ./exercise   # reads of uninitialised global memory
compute-sanitizer --tool synccheck  ./exercise   # illegal __syncthreads / warp sync
```

Run `racecheck` any time a result is *intermittently* wrong, and `memcheck`
before you ever believe a performance number -- a kernel that writes out of
bounds can be fast for reasons you will not enjoy discovering later.

## Reading SASS

```bash
nvcc -arch=native -lineinfo -o exercise main.cu
cuobjdump -sass exercise | less
nvdisasm -c -g exercise        # with control flow and line mapping
cuobjdump -ptx exercise        # the virtual ISA, closer to your source
```

What to look for, in order:

- `LDG.E.128` / `STG.E.128` - vectorised 128-bit accesses. If you wrote `float4`
  and see `LDG.E` (32-bit) instead, your alignment assumption failed.
- `LDS` / `STS` - shared memory. `LDSM` is `ldmatrix`, the tensor-core feed.
- `HMMA` / `IMMA` - tensor core instructions. Their absence in a kernel you
  believed used tensor cores is the answer to "why is it not faster".
- `LDL` / `STL` - local memory, i.e. **spills**. Find them, eliminate them.
- `BAR.SYNC` - barriers. Count them in your inner loop.
- `@!P0` predicates - predicated execution instead of branching. Cheap.
- `BSSY`/`BSYNC` - real divergence reconvergence. Not free.

## A diagnostic decision tree

```
kernel is slow
├─ sanitizer clean?           no -> fix correctness first, always
├─ SOL memory > 80%?          yes -> you are at the memory roof.
│                                    Only fusion or algorithmic change helps now.
├─ SOL compute > 80%?         yes -> use tensor cores, or lower precision,
│                                    or do less math
├─ sectors/request > 4?       yes -> coalescing. Trace warp 0's addresses.
├─ bank conflicts > 0?        yes -> shared layout. Pad or swizzle.
├─ spills > 0?                yes -> cut tile size or registers per thread
├─ achieved occupancy low?    yes -> what is the limiter? regs, smem, or blocks?
├─ Stall Barrier high?        yes -> too many syncs, or imbalance within a block
├─ Stall Long Scoreboard?     yes -> latency-bound: more in-flight loads,
│                                    unroll, double-buffer, cp.async
└─ everything looks fine but it is slow
   -> you are launch-bound or tail-bound. Check grid size vs SM count,
      check kernel duration vs 5us, go look at nsys.
```
