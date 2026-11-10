# 21 - int4 dequant GEMV

**Chapter:** [21](../../chapters/21-quantization.md)

## Task

`y = W x` with `W` stored as grouped int4. Reach **80% of your copy ceiling**
on the bytes you have to read.

There is no FLOP target because FLOPs are not the constraint and pretending
otherwise is how people waste a week here.

## Predict first

```
packed weight bytes           : ______ MB
scale + zero bytes            : ______ MB   (______% overhead)
total bytes                   : ______ MB
your measured copy ceiling    : ______ GB/s
lower bound on time           : ______ ms
arithmetic intensity          : ______ FLOP/byte
ridge point of your GPU       : ______ FLOP/byte
```

Write down the lower bound before you start. When you reach it you are done,
and you will know you are done rather than guessing.

## Run

```bash
make run
make run ARGS="16384 8192"      # M K
make run ARGS="4096 4096"       # smaller: does it still fill the GPU?
```

## Pass bar

- Within 2e-2 of the reference (which reads the original packing, dequantizes
  in fp32, and accumulates in fp32).
- >= 80% of your measured copy ceiling on packed + metadata + x + y bytes.

## The repack hook

`launch_repack` runs once, untimed, before your kernel -- exactly as a model
loader would transform weights at load time. It is the identity by default.

You may replace it with any bijective layout transformation you like, provided
your kernel reads what it produced. This is where chapter 21 exercise 21.3
becomes real: the packing format exists to serve the unpack instruction
sequence, and you are allowed to choose it.

Do this *last*. Get to 60-70% of the ceiling with the naive layout first, and
only then find out whether the unpack was ever on the critical path.

## Working order

1. One block per row, scalar loads, correct. The harness's `ref_gemv` is
   exactly that, so read it, copy its structure, and measure it. Expect
   something embarrassing; the whole exercise is the distance from here to the
   ceiling.
2. Vectorize the weight load. `uint4`, 128 bits per thread.
3. Put `x` where it belongs. It is `K` halves, read by every one of the `M`
   rows. Which memory is that?
4. Load each scale and zero once per group, not once per element.
5. Multiple rows per block, or split `K` across blocks, until the GPU is full.
6. Fix the final reduction. Shuffles.
7. Now, and only now, think about the unpack instruction count.

Re-measure after every step and compare against your predicted lower bound,
not against the previous step.

## After it passes

1. **Sweep `M`.** 512, 2048, 8192, 16384 at fixed `K`. At which `M` do you fall
   off the ceiling, and what is the fix? (This matters: real models have small
   `M` layers.)
2. **Sweep the group size.** Re-run with `GROUP` 32, 64, 256 (edit `GROUP` in
   `kernel.cu` and rebuild -- the harness follows it). Plot achieved tokens/s
   against group size and compare with your chapter 21.2 prediction.
3. **int8.** Same structure, 8 bits. Predict the time, then measure. Does the
   ratio to int4 match the byte ratio exactly? If not, what else changed?
4. **Batch it.** `x` becomes `[B, K]`. At which `B` does the kernel stop being
   bandwidth-bound and start wanting tensor cores? That number is the
   prefill/decode crossover from chapter 21.1, measured rather than predicted.
5. **Profile it.** `dram__bytes.sum` against your predicted byte count. If it
   is higher, something is being read twice -- find out what.

<details>
<summary>Hints (questions only)</summary>

- A thread issuing a 32-bit load from global memory: what fraction of peak
  bandwidth can a warp of those possibly achieve? (Chapter 03.)
- `x` is a few KB and every block reads all of it. Where should it be, and what
  does that cost you in occupancy?
- With `GROUP = 128` and one `uint4` per thread, how many weights does a thread
  hold, and how many distinct scales do they span?
- Is your reduction over `K` within a thread, within a warp, or across the
  block? Which parts of that are avoidable?
- If `M` is small, how many blocks are you launching, and how many SMs does
  your GPU have?

</details>
