# Chapter 00 - Why GPUs exist

> You cannot optimise a machine you do not have a model of. This chapter builds
> the model. No code yet. Resist the urge to skip it -- every mistake in Parts
> III and V is ultimately someone forgetting what is in here.

---

## 1. Two different problems

Imagine two jobs.

**Job A:** follow a recipe. Step 7 depends on step 6, which depends on step 5.
You cannot start step 7 early. The only way to finish faster is to do each step
faster.

**Job B:** address 10 million envelopes. No envelope depends on any other. You
could do them one at a time quickly, or you could hire 10,000 people who are
each individually slow.

A CPU is built for Job A. A GPU is built for Job B. Everything else -- warps,
coalescing, occupancy, the entire contents of this course -- follows from that
one difference in design goal.

Say it precisely:

- A CPU **minimises latency**: the time for one task to complete.
- A GPU **maximises throughput**: the number of tasks completed per second.

These are not the same goal and they trade against each other. A GPU is
*slower* than a CPU at any single thing it does. A single CUDA thread runs
roughly an order of magnitude slower than a single CPU thread. The GPU wins by
having tens of thousands of them in flight.

---

## 2. Where the transistors go

A modern CPU core spends the overwhelming majority of its transistor budget on
making one instruction stream go fast:

- **Out-of-order execution**: a window of 500+ instructions, reordered to find
  work while waiting on memory.
- **Branch prediction**: hundreds of KB of tables to guess where control flows
  next, because a mispredict costs 15-20 cycles.
- **Deep cache hierarchy**: tens of MB of L3, because a DRAM access costs
  hundreds of cycles and the only defence is not going there.
- **Speculation**: doing work that may be thrown away, on the chance it helps.

All of that machinery exists to serve *one* instruction stream. It is
spectacular engineering and it is why your laptop feels fast.

A GPU makes the opposite bet. It spends its transistors on:

- **Many simple ALUs**: thousands of them, in-order, no speculation.
- **Enormous register files**: an A100 SM has 256 KB of registers -- larger
  than its L1 cache -- so that many threads can be *resident simultaneously*
  with their state intact.
- **Wide memory interfaces**: a 5120-bit HBM bus versus a CPU's 128-bit DDR
  channel.
- **Hardware schedulers**: able to switch between resident warps with zero
  overhead, every single cycle.

The pivotal item is the third-to-last. On a CPU, a context switch means saving
registers to memory: thousands of cycles. On a GPU, every resident thread's
registers are *already* in the register file. Switching warps costs nothing
because nothing has to move.

That is the whole trick.

---

## 3. The trick, stated properly

A memory access to HBM costs ~500 cycles. A CPU fights this by trying not to
miss (caches) and by finding other work in the same instruction stream
(out-of-order execution).

A GPU does not fight it. **It accepts the latency and hides it.**

```
cycle    0: warp 0 issues a load.          stalls.
cycle    1: warp 1 issues a load.          stalls.
cycle    2: warp 2 issues some math.
cycle    3: warp 3 issues some math.
...
cycle  500: warp 0's data arrives. warp 0 is ready again.
```

As long as there is *always* at least one warp ready to issue, the ALUs never
idle, and from the outside the memory latency has vanished. It did not get
smaller. It got covered.

This reframes the central question of GPU programming. It is not "how do I make
my code fast?" It is:

> **Is there always enough independent work in flight to cover the latency?**

Every optimisation you will learn is an answer to that question or to its
companion, "am I moving more bytes than I need to?"

---

## 4. Little's Law: how much work "enough" is

This is a queueing-theory result that applies exactly, and it is the most
useful equation in this course:

```
concurrency = latency x throughput
```

To sustain a given throughput over an operation with a given latency, you must
have that much work *in flight* at all times.

Put GPU numbers in. An A100 sustains ~2000 GB/s at ~500 ns of memory latency:

```
bytes in flight = 500e-9 s x 2000e9 B/s = 1,000,000 bytes
```

**One megabyte of memory requests must be outstanding at every instant** for an
A100 to reach its rated bandwidth. If each thread has one 4-byte load pending,
that is 250,000 threads. An A100 can hold 108 SMs x 2048 threads = 221,184
threads resident. It is close, and that is not a coincidence -- the chip was
designed so that a fully occupied GPU can just barely saturate its own memory
system.

Three consequences you should carry forever:

1. **Launching too few threads cannot reach peak bandwidth.** It is arithmetic,
   not a tuning issue. A kernel on 1000 threads will never hit the roof.
2. **More bytes per thread helps.** A thread issuing a 16-byte `float4` load
   puts 4x the bytes in flight as one issuing a `float`. This is why
   vectorisation matters even though it does not change the total traffic.
3. **Independent loads per thread help.** Two loads issued back-to-back before
   either is used both count toward concurrency. This is instruction-level
   parallelism, and it is why an unrolled loop at 25% occupancy can beat a
   dependent one at 100%.

Point 3 is the one that separates people who tune occupancy from people who
understand it. We return to it in Chapter 04.

---

## 5. The hardware, floor by floor

```
GPU
 |-- Global memory (HBM / GDDR)        GBs,  ~500 cycles,  shared by everything
 |-- L2 cache                          MBs,  ~200 cycles,  shared by all SMs
 |-- SM  (Streaming Multiprocessor)  x 40-132
 |    |-- 4 warp schedulers (processing blocks)
 |    |-- Register file            64K x 32-bit (256 KB)
 |    |-- L1 / shared memory       up to 228 KB, ~25 cycles
 |    |-- CUDA cores (FP32/INT32 ALUs)      ~64-128
 |    |-- Tensor cores                      ~4
 |    |-- SFUs (transcendentals)            ~4-16
 |    |-- Load/Store units                  ~16-32
 |-- Copy engines (DMA to/from host)
```

An **SM** is the unit of scheduling and the unit you should picture. A block
lands on exactly one SM and never moves. Everything the block can do cheaply --
shared memory, `__syncthreads` -- is cheap *because* it stays on one SM.

Each SM is divided into four **processing blocks**, each with its own warp
scheduler. A scheduler picks one ready warp per cycle and issues one
instruction from it to the appropriate pipe. If no warp is ready, that cycle is
lost forever. That lost cycle is what every chapter in this course is about.

### Why 32

A **warp** is 32 threads that issue together. The number 32 is a hardware
constant on every NVIDIA GPU ever made, and it is not arbitrary: one
instruction-fetch, one decode, one scheduler slot amortised over 32 lanes is
the compromise NVIDIA settled on between control overhead (favours wide) and
divergence waste (favours narrow).

The implication is heavy. **Threads are not independent.** The thread is an
abstraction you write in; the warp is what executes. When you ask "is my kernel
coalesced?" you are asking about a warp. When you ask "does this branch cost
me?" you are asking about a warp. From here on, when thinking about
performance, think in warps.

---

## 6. SIMT: the abstraction and its leak

NVIDIA calls the model SIMT -- Single Instruction, Multiple Thread. You write
scalar code, as if for one thread:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
c[i] = a[i] + b[i];
```

The hardware runs it 32 lanes at a time. This is lovely, right up until the
lanes want to do different things:

```cuda
if (x[i] > 0) {
    y[i] = expensive_a();     // lanes 0..15 want this
} else {
    y[i] = expensive_b();     // lanes 16..31 want this
}
```

The warp cannot issue two different instructions at once. So it executes
`expensive_a` with lanes 16-31 **masked off** (they go through the motions,
their results are discarded), then `expensive_b` with lanes 0-15 masked off.

Cost = a + b, not max(a, b). Both branches, always, whenever a warp diverges.

This is the main leak in the abstraction. It is not a reason to avoid branches
-- a branch where *all 32 lanes agree* is nearly free, and that is the common
case. It is a reason to care about **which** threads take which branch. We
develop this in Chapter 05.

---

## 7. The memory wall, and why it is the real subject

Here is the number that should frighten you. Over roughly 20 years:

- Compute throughput grew by about **1000x**.
- Memory bandwidth grew by about **30x**.

The gap has to go somewhere, and it goes into the ratio you must achieve to
keep the ALUs fed. Concretely, for an A100:

```
FP32 peak      : 19.5 TFLOP/s
HBM bandwidth  : 2.0 TB/s
ratio          : 19.5e12 / 2.0e12 = ~10 FLOPs per byte
```

To use the FP32 pipes fully, every byte you read from HBM must be involved in
about 10 floating-point operations. With tensor cores in fp16 that number
becomes **~150**. On an H100 in fp16 it is **~295**.

Now price a vector add, `c[i] = a[i] + b[i]`:

```
per element: 1 FLOP, 12 bytes moved (read a, read b, write c)
intensity  : 0.083 FLOPs per byte
```

That is three orders of magnitude below what the machine needs. A vector add on
an A100 will use about **0.8% of its compute capability** no matter how
brilliant you are, and that is not a failure -- it is the correct, optimal
outcome. The kernel is memory-bound. Its job is to move bytes at 2 TB/s, and if
it does that, it is finished.

**Most kernels in machine learning are memory-bound.** Elementwise ops,
normalization, activations, softmax, attention at modest sequence length, and
*all* of LLM token generation. Matrix multiply is the famous exception, and it
is exactly why the field is organised around matrix multiply.

Internalise this now, because it determines what "optimising" means:

> For a memory-bound kernel, you do not make the math faster. You move fewer
> bytes, or you move the same bytes more efficiently. There is no third option.

Fusion, tiling, recompute, quantization, FlashAttention -- every major technique
in Parts II and V is an instance of "move fewer bytes".

---

## 8. What a GPU is bad at

Balance requires knowing the failure cases. Do not reach for a GPU when:

- **The work is serial.** A dependency chain of 1e6 steps runs at one step per
  ~500 cycles of latency. A CPU will destroy you.
- **The data is tiny.** A kernel launch costs 3-10 microseconds of wall time.
  If the work takes 2 microseconds, you have spent more on the phone call than
  on the job. (Chapter 17: CUDA Graphs exist for this.)
- **Control flow is wildly data-dependent per element.** 32 lanes going 32
  different ways costs 32x.
- **You need the answer now, once.** Throughput machines have terrible latency
  for a single small task: launch overhead, plus PCIe transfer, plus a slow
  individual thread.
- **The access pattern is pointer-chasing.** Linked lists and trees defeat
  coalescing completely; each thread's next address is unknowable until the
  previous load returns.

A good engineer knows which jobs to refuse.

---

## 9. Exercises

No code. Pencil and paper, or a calculator. Write your answers down somewhere
you can revisit -- you will want to check them again after Chapter 11.

### 0.1 - Latency vs throughput, numerically

A GPU has 100 SMs, each able to hold 2048 resident threads. A memory load takes
500 cycles. Each thread executes: one load, then 10 arithmetic instructions
that depend on it, then exits.

(a) If only one warp per SM is resident, what fraction of cycles does a warp
scheduler issue an instruction?
(b) How many resident warps per SM are needed before the schedulers are busy
every cycle, assuming each instruction issues in one cycle?
(c) What happens to that answer if each thread does 40 arithmetic instructions
instead of 10? What does that tell you about the relationship between
arithmetic intensity and the occupancy you need?

### 0.2 - Little's Law in practice

Your GPU has 900 GB/s of bandwidth and 400 ns of memory latency.

(a) How many bytes must be in flight to saturate it?
(b) You launch 100,000 threads, each with exactly one outstanding 4-byte load.
What bandwidth do you achieve?
(c) You change each thread to load a `float4` (16 bytes). Now what?
(d) Instead you keep `float` loads but unroll so each thread has 4 independent
loads in flight. Now what? Compare with (c): are they the same? Should they be?

### 0.3 - Reading a spec sheet properly

Find your GPU in [the hardware cheat sheet](../reference/hardware-cheatsheet.md)
(or run exercise `00-hello-gpu`). Compute:

(a) FLOPs per byte required to be compute-bound in fp32.
(b) The same for fp16 tensor cores.
(c) For a matrix multiply of two N x N matrices, arithmetic intensity is
roughly `N/6` when you must read both inputs from memory once (derive this; it
is the key calculation of Chapter 11). What N makes your GPU compute-bound in
fp32? In fp16 tensor cores?
(d) Transformers use hidden dimensions of 4096-8192. Are the GEMMs inside them
compute-bound on your GPU? Is the *decode* step, where the batch dimension is
1, compute-bound?

### 0.4 - Classify these

For each, state memory-bound, compute-bound, or latency-bound, and compute the
arithmetic intensity where it applies. Assume fp32 and that nothing is cached.

1. `y[i] = x[i] * 2.0f`
2. `y[i] = expf(x[i])`
3. Softmax over a row of 1024 floats
4. `C = A @ B`, all 4096 x 4096
5. `C = A @ B`, A is 1 x 4096, B is 4096 x 4096 (an LLM decode step)
6. A 3x3 convolution over a 1024 x 1024 image, 64 input and 64 output channels
7. Summing 1e9 floats to one scalar
8. `y[i] = x[idx[i]]` with random `idx` (a gather)

For (5), compare the FLOPs to (4) and then compare the *time*. The answer to
"why is LLM inference so expensive if decoding is only one token" lives here.

### 0.5 - The design question

You are designing a GPU. You can spend your transistor budget on (a) doubling
the ALUs, (b) doubling memory bandwidth, (c) doubling the register file, or (d)
doubling L2.

For each of these workloads, which do you buy, and what is the second-order
cost of your choice?

- Training a large transformer
- LLM inference at batch size 1
- Classical graphics rasterisation
- Sparse scientific computing (irregular, pointer-chasing)

There is no answer key. Write your reasoning in a paragraph each. Come back
after Chapter 21 and see whether you still agree with yourself.

---

## What you should now believe

- A GPU is a throughput machine that hides latency rather than avoiding it.
- The warp, not the thread, is the unit of execution.
- Resident parallelism is a *requirement* for bandwidth, by Little's Law, not a
  nice-to-have.
- Most ML kernels are memory-bound, and optimising them means moving fewer
  bytes.
- A kernel at 0.8% of peak FLOPs can be perfectly optimised.

Next: [Chapter 01 - The cost model](01-the-cost-model.md), where this becomes a
number you can predict before you compile.
