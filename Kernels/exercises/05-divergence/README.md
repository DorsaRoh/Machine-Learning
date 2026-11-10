# 05 - divergence

**Chapter:** [05](../../chapters/05-divergence.md)

## Task

Write a kernel whose branch pattern the harness can sweep, then explain the
resulting cost curve.

## Predict first

```
divisor   lanes taking A   predicted time vs divisor=1
   1            0                  1.00x  (baseline, no divergence)
   2           16                 ______
   4           ___                ______
   8           ___                ______
  16           ___                ______
  32           ___                ______
  64           ___                ______   <- careful here
 128           ___                ______
```

The divisor=64 row is the one to think hardest about. At divisor=64, lanes
0-31 of each even warp all take A and lanes of the odd warp all take B. Every
warp is uniform again. **Does the time return exactly to the divisor=1
baseline?** Commit to yes or no, with a reason.

## Run

```bash
make run                           # 256 LCG steps per thread
make run ARGS="4194304 2"          # a 2-step body: predication territory
make run ARGS="4194304 4096"       # a huge body: pure divergence cost
```

## Pass bar

Correctness only, exact, checked at the maximally divergent setting. The sweep
is data.

## The actual exercise

1. **Where does the curve saturate?** With a 50/50 split, divergence cost is
   2x regardless of *which* lanes take which path -- so why is divisor=2 not
   identical to divisor=16? Measure, then explain. (Hint: count how many warps
   diverge at all, not how badly.)

2. **The divisor=64 row.** Explain your measurement. If it is not exactly the
   baseline, find the residual cost. Is it the branch instructions themselves?
   Cache behaviour? Something about which *blocks* run where?

3. **Body size matters.** Compare `work=2` against `work=4096`. At `work=2` the
   divergence penalty should largely vanish. Why? Dump the SASS
   (`make sass`) for both and find the difference: look for `@P0`-style
   predicates versus `BSSY`/`BRA`/`BSYNC`. Write down which you see in each.

4. **Measure the maximum.** Modify the branch to a `switch` over
   `i % 32` with 32 different LCG constants. Predict the slowdown before you
   run it. Is it 32x? If it is less, what did the compiler do for you?

5. **The data-dependent case.** Change the branch to depend on a random array
   instead of the thread index. Sweep the probability p that a lane takes A
   from 0 to 1. Plot runtime against p. Where is the maximum, and why is the
   curve shaped like that rather than being a triangle peaking at 0.5?

6. **Now fix it.** Sort the data by the predicate before launching (you may
   just construct a pre-sorted array). Measure again at p=0.5. The speedup you
   measure is the entire business case for token sorting in MoE routing,
   Chapter 22.

<details>
<summary>Hints</summary>

- A warp diverges if its 32 lanes disagree. At divisor=16, do all 32 lanes of
  a warp disagree? How many *distinct* paths does one warp take?
- The hardware executes both sides when a warp diverges. Does it matter whether
  the split is 16/16 or 1/31? Measure it and find out.
- `work=2` means the branch bodies are a couple of instructions. What does a
  compiler do with a tiny `if/else` that has no side effects?
- Bench timing here deliberately disables the L2 flush. Why is that the right
  choice for this particular kernel, and the wrong one for `02-vector-add`?

</details>
