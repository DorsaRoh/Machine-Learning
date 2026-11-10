# 03 - stride study

**Chapter:** [03](../../chapters/03-memory-hierarchy.md)

## Task

Write one strided-read kernel. The harness sweeps the stride from 1 to 1024 and
prints the bandwidth curve for your GPU. The deliverable is not the kernel --
it is your written explanation of the curve.

## Predict first

Sketch the curve on paper before running. Mark your predictions for:

```
stride 1   : ______ GB/s   (should be near the ceiling)
stride 2   : ______        (what fraction of each sector is used?)
stride 4   : ______
stride 8   : ______
stride 16  : ______
stride 32  : ______        (what happens at exactly 32? why 32 and not 64?)
stride 64  : ______        (does it get worse than stride 32? predict yes or no)
stride 1024: ______        (any NEW effect at very large strides?)
```

Also predict the *odd* strides, 3, 6, 12, 24, 48. Should they sit on the same
curve as the even ones, or off it? There is a real reason either way.

## Run

```bash
make run
```

## Pass bar

- Correctness: exact, at stride 3 (deliberately not a power of two).
- Performance: stride 1 must reach >= 75% of the copy ceiling. The other
  strides have no bar -- they are data.

## The actual exercise

Write up answers to these. Keep the file; Chapter 16 asks you to revisit it.

1. At what stride does the curve stop getting worse? Why that stride?
2. Compute, for each stride, the number of 32-byte sectors a warp touches.
   Multiply by 32 bytes to get hardware traffic. Divide useful bytes by that
   and compare to your measured ratio. How close is the model?
3. At the plateau, what fraction of the measured ceiling remains? It is
   probably not exactly 1/8. Name two reasons the model overestimates the
   damage.
4. At stride 1024 and beyond, is there a *further* drop? If so, what structure
   besides the cache is now being missed? (Consider what maps virtual addresses
   to physical pages.)
5. The write is contiguous in every run. How much of the measured bandwidth at
   stride 32 is the write, and how much is the read? Design a way to separate
   them, then do it.

## Verify with the profiler

```bash
make profile
```

Find `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` and the corresponding
request count. The ratio is sectors per request. Confirm it is 4 at stride 1
and 32 at stride 32. If your hand count and the counter disagree, the counter
is right and your model has a gap worth finding.

<details>
<summary>Hints</summary>

- A sector is 32 bytes = 8 floats. How many distinct sectors do 32 lanes touch
  at stride 2? At stride 8? At stride 9?
- Why would an odd stride behave differently from an even one of similar size?
  Think about which sector each lane lands in, not just how far apart they are.
- The curve flattens when each lane is already alone in its sector. Adding more
  distance cannot make that worse. What *can* still get worse?
- At stride 1024 with 4-byte floats, how many bytes apart are consecutive
  lanes? How does that compare to a 4 KB page? How many TLB entries does one
  warp now need?

</details>
