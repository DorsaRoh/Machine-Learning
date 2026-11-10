# 00 - hello-gpu

**Chapters:** [00](../../chapters/00-why-gpus-exist.md), [01](../../chapters/01-the-cost-model.md)

## Task

Three lines of work in `kernel.cu`:

1. `global_index()` must return a unique index per thread across the whole grid.
2. `write_indices` must not write out of bounds when `n` is not a multiple of
   the block size.
3. `launch_write_indices` must launch enough blocks to cover all `n` elements.

`n` defaults to 1000003, which is prime. That is not decoration: it guarantees
no block size divides it evenly, so a missing bounds guard corrupts memory and
a wrong grid size leaves the tail unwritten. Both failures are visible in the
gate output.

## Predict first

Before running, write down:

- With `block = 256` and `n = 1000003`, how many blocks do you need?
- How many threads does that launch? How many of them have no work?
- Memory is poisoned to `-1` before the launch. If your grid is one block too
  small, which index is the first to still read `-1`?

## Run

```bash
make run
make run ARGS=1000000    # a round n, to compare
```

## Pass bar

Correctness only, exact. No performance gate -- this exercise exists to prove
your toolchain works and to produce the device report.

## After it passes

Copy the measured copy bandwidth into the roofline worksheet the harness
prints, and finish Exercise 1.2 from Chapter 01 using *your* machine's numbers
rather than the A100 used in the examples. Keep that page. Every performance
gate in this course is a percentage of the number you just measured.

Also worth noticing: the measured copy bandwidth is below the datasheet figure.
Before Chapter 03, write one sentence predicting where the missing percent
went.

<details>
<summary>Stuck? (questions, not answers)</summary>

- If every block computes `threadIdx.x`, how many distinct values does the
  whole grid produce? How many do you need?
- What identifies a block, and how many threads came before it?
- Integer division truncates. If you launch `n / block` blocks and `n` is not
  divisible by `block`, which elements never get a thread?
- Who writes to `out[i]` when `i >= n`? What is at that address?

</details>
