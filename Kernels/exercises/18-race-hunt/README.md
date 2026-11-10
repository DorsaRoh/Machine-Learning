# 18 - race hunt

**Chapter:** [18](../../chapters/18-memory-model.md)

## Task

Four kernels, one synchronisation bug each. Find them, classify them, fix them
without changing what they compute.

Most of them will produce the right answer on your first run. That is the
lesson, not an inconvenience.

## For each bug, write down

```
bug N:
  what is unsynchronised: _______________________________________
  when does it manifest:  _______________________________________
    (more threads? more blocks? a different GPU? under load?)
  which sanitizer finds it: memcheck / racecheck / initcheck / synccheck / none
  the fix:                _______________________________________
  why the fix is sufficient (not just "it passes now"): __________
```

Not all four are caught by a tool. Before you run anything, read what
`racecheck` actually inspects -- which memory space, and which hazards -- then
predict for each bug whether it will be reported. Check your predictions. The
bugs no tool can see are the most important ones here, because in real code
they are the ones that survive to production.

## Run

```bash
make run              # 500 runs of each
make run ARGS=5000    # when you want to be more certain
make sanitize         # memcheck then racecheck
compute-sanitizer --tool synccheck ./exercise
```

## Pass bar

- All four fixed kernels: 0 bad runs out of 500, and bit-identical output
  across runs.
- `make sanitize` clean on the fixed versions.

The buggy versions may well show 0 bad runs too. Do not let that persuade you
that there is no bug.

## Notes on each

**bug_1** loops over tiles, reusing the same shared array. Chapter 06 section 2
describes the required barrier discipline for exactly this shape.

**bug_2** has 4096 blocks and 8 accumulators, so 512 blocks hit each one. The
per-block reduction is fine. Look at the last three lines. Note that the values
are small integers, so a correct version is bit-reproducible regardless of the
order blocks finish in -- if your fixed version is unstable across runs, that
is information too.

**bug_3** is copied from a pre-2017 tutorial, where it was correct. Chapter 05
section 2 explains what changed in 2017. The fix is not to add barriers back.

**bug_4** uses `volatile` and looks careful. Chapter 18 sections 3 and 4
explain what `volatile` does and does not provide. It has more than one
problem; find all of them, including one that could deadlock.

## After it passes

1. **Make bug_1 fail reliably.** Increase `tiles`, increase block count, add
   contention. What makes the failure rate go up? Being able to *provoke* a
   race is as valuable as fixing one.
2. **bug_3 across architectures.** If you have access to both a Turing and an
   Ampere/Hopper GPU, run the buggy version on each. Does the failure rate
   differ?
3. **bug_4, properly.** Rewrite it three ways: with `__threadfence`, with
   `cuda::atomic` acquire/release, and with two kernel launches. Compare SASS
   for the first two. Measure all three. Which would you ship, and why is the
   boring answer often right?
4. **Audit your own work** (Chapter 18 exercise 18.5). Run memcheck and
   racecheck on every exercise from Parts I-III. Report what you find. Bugs
   that were present but silent are the ones to think about hardest.

<details>
<summary>Hints (questions only)</summary>

- bug_1: in iteration `t+1`, thread 5 writes `s[5]`. In iteration `t`, which
  threads read `s[5]`? Have they finished?
- bug_2: `out[g] += s[0]` is three operations, not one. Write them out. Now
  interleave two blocks' versions of those three operations and see what you
  lose.
- bug_3: on Volta and later, are the 32 lanes of a warp guaranteed to execute
  in lockstep between those six lines? What would make the exchange between
  lanes safe, and is shared memory even the right mechanism here?
- bug_4: (i) who guarantees `*scale` is visible once `*ready` is 1? (ii) only
  thread 0 of each block waits -- what are the other 255 doing? (iii) if the
  grid has more blocks than can be resident, what happens to a block spinning
  on block 0's flag?

</details>
