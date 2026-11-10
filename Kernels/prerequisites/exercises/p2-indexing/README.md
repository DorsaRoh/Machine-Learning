# p2-indexing

**Chapter:** [P3 - Arrays and indexing](../../p3-arrays-and-indexing.md)

> If you do only one exercise in the prerequisites, do this one — and do it
> until it is boring. Every kernel in the main course is this arithmetic with a
> thread index substituted for a loop variable.

## Task

Flattening, unflattening, strides, 4D tensors, sub-matrix views, transposes,
bounds-clamped tiles, and a timed demonstration of why loop order decides
performance. Ninety minutes, and worth every one of them.

## Run

```bash
make run
make debug        # when a tile index goes off the end, this tells you where
```

## Pass bar

All 39 checks pass. The timing at the end is not gated — it is there to be
predicted.

## Notes

- **Part 1** is one line per function. If any of them takes you more than a
  line, re-read P3 section 2.
- **Part 4's `index_4d`** has a deliberate trap: one of `B, H, S, D` is never
  needed. Work out which before you start typing. The fact that you never need
  the outermost extent is why a kernel can be handed a pointer into the middle
  of a tensor and work correctly on it.
- **Part 5's `load_tile`** is the `if (row < M && col < N)` guard that appears
  in every kernel from Chapter 06 onward. The checks specifically probe the
  ragged edge: tile (1,1) of a 20x20 matrix is mostly outside the matrix.
  Getting that wrong is a buffer overrun on a CPU and silent garbage on a GPU.
- **Part 6** does the same 16 million multiplications twice.

## Predict before you run

Before `make run` succeeds, write down your answer: how many times slower is
the column-major loop order going to be, on 4096x4096 floats?

Most people say "2x, maybe 3x — it's the same number of operations". On the
machine this exercise was written on, it is **64x**. Sixty-four times, for
reordering two loops. If your prediction was off by more than 3x, go back to
P3 section 5 and P0 section 4 and work out where your mental model broke.

That gap, between "same arithmetic" and "64x", is the entire reason this
course exists.

## After it passes

1. Re-run part 6 at `N = 512` (1 MB, fits in cache) and at `N = 8192`. How does
   the ratio change with size? Why would a small matrix show a smaller gap?
2. Write a blocked transpose: instead of one element at a time, move 32x32
   tiles. Does it beat your naive transpose? By how much? You have just
   previewed Chapter 06 — and the reason the GPU version needs shared memory
   is the same reason your blocked version needs the tile to fit in L1.
3. Implement `index_4d` for the *other* common layout, `[B, S, H, D]`, which is
   what you get before a transpose in an attention implementation. Which
   stride changed?
4. Take `load_tile` and remove the bounds check. Run under `make debug`. Save
   the error message; it is what a GPU out-of-bounds write looks like, except
   that on a GPU nothing tells you.
5. Do P3.6 on paper: write the index expressions for a batched matmul,
   `C[b] = A[b] @ B[b]`.

<details>
<summary>Hints (questions only)</summary>

- `row_of`/`col_of`: you have one integer and need two. What two operators take
  a number and a divisor and together lose no information?
- `flat_index_colmajor`: in column-major, which neighbour is adjacent in
  memory — the one to the right, or the one below? So which index gets
  multiplied?
- `transpose`: the output has shape `cols x rows`. When you write
  `out[? * ? + ?]`, which extent is the row length of `out`?
- `index_4d`: start from the innermost dimension and work out. `d` moves by 1.
  How far does `s` move? Then `h`? Then `b`? Do you ever need to know how many
  batches exist?
- `load_tile`: what are the matrix coordinates of element `(i, j)` of tile
  `(ti, tj)`? Which two comparisons decide whether that cell exists?
- Part 6: in each version, how far apart in memory are two consecutive
  iterations of the *inner* loop? How big is a cache line (P0 section 4)? How
  many of the 64 bytes you fetched do you actually use before evicting it?

</details>
