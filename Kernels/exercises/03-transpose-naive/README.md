# 03 - naive transpose

**Chapter:** [03](../../chapters/03-memory-hierarchy.md) (fixed in
[06](../../chapters/06-shared-memory-tiling.md))

## Task

Write the obvious transpose. Then explain its cost.

## Predict first

For 8192 x 8192 fp32 on your GPU:

```
compulsory bytes        = ______        (read every element once, write once)
best possible time      = ______ ms     (at your measured ceiling)
sectors read per warp   = ______
sectors written per warp= ______
predicted slowdown vs copy = ______x
predicted time          = ______ ms
```

The harness measures a same-shape copy for you, so you can compare against a
local reference rather than a global ceiling.

## Run

```bash
make run
make run ARGS=256     # small enough to fit in L2 -- note what happens
make run ARGS=8192
```

## Pass bar

Correctness only, exact. There is deliberately no performance gate: this kernel
cannot be made fast without shared memory, and pretending otherwise would teach
you to flail.

## The actual exercise

1. **Account for the slowdown.** Your predicted factor vs the measured one,
   within 20%. If you are outside that, find the missing effect. (Two
   candidates: the L2 is catching some of the strided writes, and write-allocate
   is inflating the traffic. Which is it? Prove it with a counter.)

2. **Flip the mapping.** Try `out[x*n+y] = in[y*n+x]` and `out[y*n+x] =
   in[x*n+y]`. One has a coalesced read and a strided write; the other the
   reverse. Measure both. Are they the same speed? Should they be? Think about
   whether the memory system treats a scattered read and a scattered write
   symmetrically, and what a write-allocate does to the asymmetry.

3. **Run at n=256.** The matrix is 256 KB. Compare with n=8192 and explain the
   difference in one sentence. This is the L2 benchmarking lie from Chapter 03
   section 7, and seeing it once will stop you being fooled later.

4. **Block shape.** The harness launches 32x8. Try 32x32, 16x16, 8x32. Which is
   best, and can you predict it from the warp layout rather than by trying?

5. **Argue that indexing cannot fix this.** Write a paragraph proving that no
   assignment of threads to elements makes both the read and the write
   coalesced. The structure of that proof is exactly why shared memory exists,
   and it is the setup for Chapter 06.

<details>
<summary>Hints</summary>

- In a 32x8 block, which threads form warp 0? Do they vary in x or in y?
- For warp 0, write out the 32 read addresses and the 32 write addresses. How
  many sectors each?
- A strided write and a strided read both move 32 sectors. But one of them may
  additionally have to *fetch* those sectors before modifying them. Which?
- At n=256 the entire matrix is smaller than your L2. What is the strided
  access actually hitting?

</details>
