# 25 - ring AllReduce

**Chapter:** [25](../../chapters/25-multi-gpu.md)

## Task

Implement ring AllReduce with `cudaMemcpyPeerAsync` and reach 60% of NCCL at
64 MB.

**This exercise needs 2 or more GPUs.** If you have one, the harness says so
and exits cleanly. Rent a 2x or 4x instance for an hour -- it costs a few
dollars and it is the only way to have actually measured an interconnect,
which is the real point of chapter 25.

## Predict first

Run the harness once to get the topology, then:

```
GPUs                           : ______
link type (p2p / pci)          : ______
measured P2P bandwidth         : ______ GB/s
bytes a ring moves per GPU, 64 MB buffer : ______ MB
predicted time at that bandwidth         : ______ ms
predicted algorithm bandwidth            : ______ GB/s
per-hop latency                : ______ us
steps                          : ______
latency floor                  : ______ us
size at which latency == bandwidth time  : ______ bytes
```

That last number predicts the shape of the whole table the harness prints. Check
it afterwards.

## Run

```bash
make run
make run ARGS=2       # use only 2 of the available GPUs
```

Needs NCCL for the baseline. Without it the harness runs correctness only and
tells you the gate was skipped.

## Pass bar

- Exact on every GPU (the inputs are small integers; the sum is exact in fp32,
  so "close" means wrong).
- >= 60% of NCCL's algorithm bandwidth at 64 MB.

## How to get there

Start with the version that synchronizes every device after every step. It is
obviously correct and it will be slow. Then remove synchronizations one at a
time, re-verifying after each removal. A ring that is wrong only occasionally
is the easiest thing in this course to convince yourself is right.

Watch for the two hazards, which are chapter 18's hazards one level up:

- A GPU adding a chunk that has not finished arriving.
- A GPU overwriting a chunk its neighbour is still reading.

`cudaEventRecord` / `cudaStreamWaitEvent` across devices expresses both without
a full device sync. Find out what the full sync was costing you before you
replace it.

## Reading the table

The harness sweeps 1 KB to 64 MB. You should see:

- **Small sizes:** both you and NCCL flat in *time*, rising in GB/s. That is
  the `2(N-1) * latency` term, and nothing about bandwidth matters there.
- **Large sizes:** both approaching a bandwidth asymptote. This is where your
  60% is judged.
- **A knee** where the two terms cross. Compare it to your prediction.

If your small-size numbers are much worse than NCCL's, you have too many
synchronizations. If your large-size numbers are much worse, you are not
keeping the link busy -- how many transfers do you have in flight at once?

## After it passes

1. **Vary the GPU count.** 2 and 4. Does your time per byte change? Should it?
   (Section 2, consequence 1.)
2. **Direct peer loads.** Replace `cudaMemcpyPeerAsync` with a kernel that
   reads the neighbour's memory directly and adds. Faster? At which sizes, and
   why not at the others?
3. **Bucketing.** Chapter 25 exercise 25.4: 200 tensors totalling 400 MB,
   one at a time versus 25 MB buckets versus one buffer. Plot it.
4. **Tree.** Implement a tree AllReduce. At which size does it beat your ring,
   and does the crossover match the `log N` versus `N` latency argument?
5. **Profile.** `nsys profile --trace=cuda,nvtx` and look at the per-device
   timeline. Are the links busy simultaneously, or are you using one at a time?

<details>
<summary>Hints (questions only)</summary>

- In reduce-scatter step `s`, GPU `g` sends which chunk index? Write it out for
  4 GPUs, 3 steps, and check that after step 2 each GPU owns exactly one fully
  reduced chunk.
- Where does the incoming chunk land? Can it land directly on top of the
  destination's chunk, or does it need the scratch buffer?
- `cudaMemcpyPeerAsync` is issued on a stream belonging to which device?
- How many of the `ngpu` links are busy at once in your implementation? How
  many could be?
- What does `cudaDeviceSynchronize` on GPU 0 do to GPU 1's work?

</details>
