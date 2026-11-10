# Chapter 25 - Multi-GPU

> Everything so far assumed one GPU and one memory. Add a second GPU and you
> add a new level to the memory hierarchy -- one that is 10x to 100x slower
> than HBM -- plus a new source of stalls that no single-GPU profiler will show
> you. The roofline still works. The numbers in it change.

---

## 1. The new rung of the hierarchy

Extend the Chapter 03 table:

| Level | Bandwidth | Latency |
|---|---|---|
| Registers | ~100 TB/s | ~0 |
| Shared / L1 | ~20 TB/s | ~30 cycles |
| L2 | ~5 TB/s | ~200 cycles |
| HBM | 1-3 TB/s | ~400-800 cycles |
| NVLink (per GPU, aggregate) | 300-900 GB/s | ~2-10 us |
| PCIe 4 x16 | ~25 GB/s | ~5-20 us |
| InfiniBand / Ethernet | 25-50 GB/s | ~2-10 us |

Two things to read off it. NVLink is roughly one third of HBM bandwidth, so
inter-GPU traffic is expensive but not catastrophic. **PCIe is 50x slower than
HBM**, so a machine without NVLink is a fundamentally different machine, and a
strategy that works on a DGX can be unusable on a workstation.

Check which you have before designing anything:

```bash
nvidia-smi topo -m        # NV# = NVLink, PIX/PHB/SYS = PCIe, decreasing goodness
```

---

## 2. Collectives, and what they cost

NCCL implements the standard set. The two that matter most in training:

**AllReduce** -- every GPU contributes a tensor, every GPU ends with the sum.
This is gradient synchronization in data-parallel training, and it is usually
the dominant communication cost.

**AllGather / ReduceScatter** -- the two halves of an AllReduce, and the
primitives of FSDP and ZeRO. `AllReduce = ReduceScatter + AllGather`, which is
exactly how ring AllReduce is implemented.

**Ring AllReduce**, the algorithm worth knowing in detail because its cost
model explains almost everything:

```
N GPUs in a logical ring, tensor split into N chunks.
Phase 1 (reduce-scatter): N-1 steps; each step every GPU sends one chunk to
        its neighbour and accumulates the chunk it receives.
        After N-1 steps each GPU owns the fully reduced version of one chunk.
Phase 2 (all-gather):     N-1 steps; the reduced chunks circulate.

bytes sent per GPU = 2 * (N-1)/N * S     for a tensor of S bytes
time               = 2 * (N-1)/N * S / BW  +  2 * (N-1) * latency
```

Three consequences:

1. **The bandwidth term is independent of `N`** (it approaches `2S/BW`). Ring
   AllReduce scales to many GPUs without the per-GPU traffic growing. This is
   why it won.
2. **The latency term is linear in `N`.** For small tensors, `2(N-1)` hops of
   ~5 us dominates completely. This is why you bucket small gradients together
   before reducing them.
3. **Tree AllReduce** has a `log(N)` latency term and worse bandwidth
   behaviour, so NCCL picks tree for small messages and ring for large ones,
   automatically. Knowing why lets you read its choice in the logs.

---

## 3. Overlapping communication with compute

This is the whole game, and it is Chapter 17 applied at a larger scale.

In data-parallel training, gradients become available during the backward pass,
layer by layer, from the output backwards. You do not have to wait for the
whole backward to finish before you start reducing:

```
backward layer L   -> gradient ready -> start AllReduce on a separate stream
backward layer L-1 -> gradient ready -> start AllReduce
...
wait for all AllReduces just before the optimizer step
```

With enough compute left in the backward to cover the communication, the
AllReduce is free. PyTorch DDP does this automatically, bucketing gradients
into ~25 MB groups: large enough to be bandwidth-bound rather than
latency-bound, small enough to start early.

The arithmetic you should be able to do: given model size `S`, backward time
`T_b` and link bandwidth `BW`, communication is hidden if
`2 * S / BW < T_b`. When it is not, no amount of overlap saves you and you need
a different parallelism strategy. Exercise 25.2 works this through.

**The thing that silently breaks overlap** is the same as in Chapter 17: a
synchronization you did not know was there. `.item()`, `print(loss)`,
`tensor.cpu()`, a Python conditional on a tensor value -- each forces a device
sync and drains the overlap. In a training loop, logging every step is often
the reason the GPUs are 70% utilized.

---

## 4. The parallelism strategies, in one page

**Data parallel.** Replicate the model, split the batch, AllReduce gradients.
Communication per step is `2S` bytes regardless of batch size, so it scales
well with large batches and badly with small ones. Needs the model to fit on
one GPU.

**Tensor parallel.** Split individual matrices across GPUs. A transformer MLP
splits column-wise then row-wise, needing one AllReduce per layer in forward
and one in backward -- so communication is *inside* the critical path and
cannot be fully overlapped. Only viable over NVLink; over PCIe it is a
disaster. This is the one that most directly becomes a kernel problem, because
the fused "GEMM + AllReduce" kernels that hide the communication inside the
matmul epilogue are exactly the kind of thing you now know how to write.

**Pipeline parallel.** Split layers across GPUs, feed micro-batches. Tiny
communication (one activation per boundary) but introduces bubbles; the bubble
fraction is `(P-1)/(M+P-1)` for `P` stages and `M` micro-batches, which tells
you how many micro-batches you need.

**FSDP / ZeRO.** Shard parameters, optimizer state and gradients. AllGather
parameters just before use, ReduceScatter gradients after. Trades more
communication for much less memory, and the prefetch of the next layer's
parameters is, once again, the Chapter 17 pipeline.

Real systems compose all four. The thing to carry away is that each one has a
communication pattern whose cost you can compute, and the choice is always
"which of these fits in the gaps my compute leaves".

---

## 5. Writing it yourself

You will rarely beat NCCL, but implementing a ring AllReduce once makes the
cost model concrete.

Peer-to-peer access, when the topology allows:

```cuda
cudaDeviceCanAccessPeer(&can, dev_a, dev_b);
cudaSetDevice(dev_a); cudaDeviceEnablePeerAccess(dev_b, 0);
cudaMemcpyPeerAsync(dst, dev_b, src, dev_a, bytes, stream);
```

With peer access enabled, a kernel on GPU A can dereference a pointer into
GPU B's memory directly, which is how the fastest custom collectives work: no
staging copy, just loads and stores across NVLink from inside a kernel. The
costs are what you would expect -- the latency is much higher and you need many
more concurrent accesses in flight (Little's Law, Chapter 01) to saturate the
link.

The synchronization is the hard part. Cross-GPU, you need the Chapter 18
machinery one level up: flags in peer memory, release/acquire semantics, and an
ordering argument you can actually defend. `cuda::atomic` with
`system` scope, or NCCL's own primitives, rather than `volatile` and hope.

---

## 6. Common failure modes

**Scaling is 1.6x on 2 GPUs instead of 1.95x.** Communication is not
overlapping. Look at an `nsys` timeline with `--trace=cuda,nvtx,nccl` and find
whether the NCCL kernels run alongside compute or between it.

**Fine on 2 GPUs, terrible on 8.** You are latency-bound; your messages are too
small. Bucket.

**Fine on the DGX, terrible on the workstation.** PCIe instead of NVLink.
Check `nvidia-smi topo -m`. The fix is usually a different parallelism
strategy, not a faster kernel.

**A hang.** The classic: ranks disagree about which collective to call, or
about its size. Collectives must be called in the same order with the same
shapes on every rank. A conditional that depends on local data will deadlock
eventually.

**Utilization at 70% with no obvious gap.** Look for `.item()`, `print`, or
`assert` on a tensor in the loop.

**Loss diverges only in multi-GPU.** Gradient reduction averaging wrong (sum
versus mean), or a seed shared where it should differ, or a rank not
participating.

---

## 7. Exercises

### 25.1 - Know your machine

(a) `nvidia-smi topo -m`. Draw the topology.
(b) Measure P2P bandwidth for every pair (`p2pBandwidthLatencyTest` from the
    CUDA samples, or your own).
(c) Measure P2P latency for a 4-byte transfer.
(d) Compare both to HBM. Where does this rung sit?
(e) If you have only one GPU: do this on a rented 2x or 4x instance for an
    hour. It is worth the few dollars to have measured it once.

### 25.2 - The overlap budget

For a 1B-parameter model, fp16 gradients, on your interconnect:

(a) Bytes AllReduced per step.
(b) Ring AllReduce time at your measured bandwidth, for 2, 4, 8 GPUs.
(c) Backward pass time (measure it).
(d) Is communication hideable? At which GPU count does it stop being?
(e) Repeat for a 70B model. What changes qualitatively?
(f) Repeat over PCIe instead of NVLink. Same question.

### 25.3 - Implement ring AllReduce

(a) Implement it with `cudaMemcpyPeerAsync` and a reduction kernel.
(b) Verify against NCCL for correctness.
(c) Measure both across sizes from 1 KB to 1 GB. Plot the ratio.
(d) At which size are you closest to NCCL? At which are you worst?
(e) Explain the small-message gap in terms of the latency term in section 2.
(f) Now implement it with direct peer loads inside one kernel instead of
    `memcpyPeer`. Does it help? At which sizes?

### 25.4 - Bucketing

(a) Take 200 gradient tensors of varying sizes totalling 400 MB.
(b) AllReduce them one at a time. Measure.
(c) Bucket into 25 MB groups. Measure.
(d) One single 400 MB buffer. Measure.
(e) Plot time against bucket size. Where is the optimum, and which term in the
    cost model does each side of the curve correspond to?

### 25.5 - Find the stall

Take a 2-GPU DDP training loop:

(a) `nsys profile` with NCCL tracing. Where are the gaps?
(b) Add a `loss.item()` to the loop. Re-profile. What happened?
(c) Remove it, add gradient accumulation over 4 steps. Re-profile.
(d) What is your actual scaling efficiency, and where is the missing time?

---

## Exercises to run

- [`25-allreduce`](../exercises/25-allreduce) - ring AllReduce by hand.
  Bar: exact, >= 60% of NCCL at 64 MB. (Needs 2+ GPUs; rent them for an hour.)

---

## What you should now believe

- The interconnect is another rung on the memory hierarchy, and the roofline
  method works unchanged once you put its bandwidth in.
- Ring AllReduce moves `2(N-1)/N * S` bytes per GPU -- bandwidth independent of
  `N`, latency linear in `N`. That single formula explains bucketing, tree
  algorithms, and most scaling failures.
- Overlapping communication with compute is Chapter 17 at a larger scale, and
  it is broken by the same thing: an accidental synchronization.
- The parallelism strategy is chosen by whether its communication fits in the
  gaps your compute leaves. That is a calculation, not a preference.
- NVLink versus PCIe is not a 2x difference; it changes which strategies are
  possible at all.

Next: [the capstone](26-capstone.md).
