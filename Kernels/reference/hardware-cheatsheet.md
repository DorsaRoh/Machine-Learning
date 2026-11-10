# Hardware cheat sheet

Numbers you should eventually hold in your head, because they let you reject
bad designs before writing them.

> Always prefer what `bench::device_info()` reports on the machine in front of
> you. Vendors ship many variants of the same name (A100 40GB vs 80GB differ by
> 50% in bandwidth) and clocks throttle under load.

## The GPUs you will actually use

| | T4 | A10G | A100 80GB | H100 SXM | RTX 4090 |
|---|---|---|---|---|---|
| Arch / `sm_` | Turing 75 | Ampere 86 | Ampere 80 | Hopper 90 | Ada 89 |
| SMs | 40 | 80 | 108 | 132 | 128 |
| HBM/GDDR bandwidth | 320 GB/s | 600 GB/s | 2039 GB/s | 3350 GB/s | 1008 GB/s |
| L2 | 4 MB | 6 MB | 40 MB | 50 MB | 72 MB |
| Shared mem / SM | 64 KB | 128 KB | 164 KB | 228 KB | 100 KB |
| Max shared / block | 64 KB | 99 KB | 163 KB | 227 KB | 99 KB |
| Registers / SM | 64 K | 64 K | 64 K | 64 K | 64 K |
| Max threads / SM | 1024 | 1536 | 2048 | 2048 | 1536 |
| FP32 peak | 8.1 TF | 31 TF | 19.5 TF | 67 TF | 82 TF |
| TF32 tensor | -- | 63 TF | 156 TF | 495 TF | 83 TF |
| FP16 tensor | 65 TF | 125 TF | 312 TF | 990 TF | 165 TF |
| FP8 tensor | -- | -- | -- | 1979 TF | 330 TF |
| `cp.async` | no | yes | yes | yes | yes |
| TMA / clusters | no | no | no | yes | no |
| Roofline knee (fp32) | ~25 F/B | ~52 F/B | ~10 F/B | ~20 F/B | ~81 F/B |
| Roofline knee (fp16 TC) | ~203 F/B | ~208 F/B | ~153 F/B | ~295 F/B | ~164 F/B |

Tensor-core figures are dense (non-sparse) peaks. Read the knee column as: "a
kernel must do at least this many FLOPs per byte of HBM traffic before compute
can possibly be the bottleneck." Note how brutal it is for tensor cores -- an
H100 needs ~295 FLOPs per byte in fp16. This single row explains why nearly
every modern optimisation is about *not moving bytes*.

## Latencies, in cycles

| Operation | Approx. cycles | Note |
|---|---|---|
| Register access | ~1 | free, in practice |
| Shared memory, no conflict | 20-30 | the reason tiling works |
| Shared memory, 32-way conflict | 600+ | serialised, as bad as a cache miss |
| L1 hit | ~30 | |
| L2 hit | ~200 | |
| HBM / global | 400-800 | what occupancy exists to hide |
| FMA fp32 | ~4 (pipelined) | throughput 1/cycle/core |
| Transcendental (`__expf`, `rsqrtf`) | ~4-8, SFU | 1/4 rate of FMA -- cheap, but count them |
| `exp` (precise) | ~20+ | a software routine; `__expf` is hardware |
| Atomic to same address (shared) | serialised | 32-way contention = 32x |
| Kernel launch | 3-10 us wall | why sub-10us kernels need CUDA Graphs |

## Fixed structural facts

- Warp = 32 threads. Has been for every NVIDIA GPU ever shipped.
- Max threads per block = 1024.
- Shared memory = 32 banks x 4 bytes wide.
- Memory transactions are 32-byte sectors; a cache line is 128 bytes.
- Max registers per thread = 255. Hit that and you will spill.
- Grid dimensions: x up to 2^31-1, y and z up to 65535.
- Blocks per SM: hardware limit is 16-32 depending on architecture, and you
  will usually hit a shared-memory or register limit long before it.

## Occupancy arithmetic you can do in your head

Resident warps per SM is the **minimum** of four limits:

```
1. hardware cap        : max_threads_per_SM / 32
2. register limit      : regs_per_SM / (regs_per_thread * 32)
3. shared memory limit : (smem_per_SM / smem_per_block) * warps_per_block
4. block-count limit   : max_blocks_per_SM * warps_per_block
```

Worked example, A100, a kernel using 64 registers/thread and 48 KB shared with
256-thread blocks:

```
hardware  : 2048/32 = 64 warps
registers : 65536 / (64*32) = 32 warps      <-- binding constraint
shared    : (164/48) = 3 blocks * 8 warps = 24 warps   <-- actually this one
blocks    : 32 * 8 = 256 warps
=> 24 of 64 warps = 37.5% occupancy
```

Two things to notice: the binding constraint is rarely the one you assumed, and
37.5% is not automatically bad. A register-rich, shared-memory-rich kernel is
usually one with high arithmetic intensity, which needs fewer warps to stay
busy. Chapter 04 makes this precise.

## Converting counters to bytes

```
bytes from HBM  = dram__bytes.sum
sectors         = l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
requests        = l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
sectors/request = how many 32-byte chunks each warp instruction touched
```

`sectors/request` is the coalescing meter. **4 is perfect** for 32 threads
reading 4-byte values (32 x 4 = 128 bytes = 4 sectors). **32 is the worst
case** and means you are moving 8x the bytes you need.
