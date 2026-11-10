# P3 - Arrays, indexing and strides

> This is the most important chapter in the prerequisites, and it is not close.
> Chapter 03's coalescing, Chapter 06's tiling, Chapter 11's matrix multiply
> and Chapter 19's attention are all the same skill: knowing which flat index
> holds the element you want, and how far the index moves when you move.
>
> Work until this is automatic. Not "I can work it out" — automatic.

---

## 1. Memory is flat, your data is not

You want a 4x6 matrix. Memory gives you a line of bytes. So somebody has to
decide how to lay a rectangle out in a line, and everyone has agreed on this:

**Row-major: store row 0, then row 1, then row 2, ...**

A 3x4 matrix (3 rows, 4 columns):

```
logical:                     flat array (12 elements):

      c0   c1   c2   c3
 r0 [  a    b    c    d  ]    [ a b c d | e f g h | i j k l ]
 r1 [  e    f    g    h  ]      0 1 2 3   4 5 6 7   8 9 ...
 r2 [  i    j    k    l  ]      <-- r0 --> <-- r1 --> <- r2 ->
```

The formula, which you must know cold:

```
index = row * num_cols + col
```

Check it: row 1, col 2 → `1 * 4 + 2` = 6 → `g`. Correct.

**Note what the formula uses:** `num_cols`, the width. Not the number of rows.
The width is what tells you how far to jump to get to the next row, and the
height never enters into it. If you find yourself reaching for `num_rows` in an
index calculation, stop and recheck.

In C, a declared 2D array does this for you:

```cpp
float m[3][4];
m[1][2] = 5.0f;        // the compiler computes 1*4+2 for you
```

But the moment data comes from `malloc`, `new`, `cudaMalloc`, PyTorch, or a
file, it arrives as a flat pointer and you do the arithmetic yourself:

```cpp
float* m = new float[3 * 4];
m[1 * 4 + 2] = 5.0f;   // you compute it
```

**All GPU memory is flat.** `cudaMalloc` returns a `float*`. There is no
`float m[3][4]` on the device side of this course. So the flat form is not an
alternative — it is the only form.

---

## 2. Going backwards: flat index to coordinates

Just as often you have a flat index and need the coordinates. This happens in
*every* CUDA kernel, because the GPU hands each thread a single number and the
thread has to work out which element it owns.

```
row = index / num_cols        // integer division
col = index % num_cols        // remainder
```

Index 6 in a 4-wide matrix: `6 / 4 = 1`, `6 % 4 = 2` → row 1, col 2. Matches.

This is why P1 made a point of integer division and `%`. They are not trivia;
they are the decomposition operators.

The two directions are exact inverses, and you should be able to go either way
without thinking:

```
flatten:   index = row * W + col
unflatten: row = index / W,  col = index % W
```

---

## 3. Strides: the general form

**A stride is how far the flat index moves when one coordinate moves by one.**

For a row-major `H x W` matrix:

- Move one column (`col += 1`) → index moves by **1**. Column stride = 1.
- Move one row (`row += 1`) → index moves by **W**. Row stride = W.

So the formula generalises:

```
index = row * row_stride + col * col_stride
      = row * W          + col * 1
```

Why bother with the general form? Because it covers cases the simple formula
does not, and those cases are everywhere in real code:

**Column-major** (Fortran, MATLAB, and cuBLAS) stores columns first:

```
index = row * 1 + col * H          row stride 1, col stride H
```

This is not a curiosity. **cuBLAS is column-major**, and PyTorch is row-major,
which is why the harnesses in Chapters 11, 19 and 20 do the "swapped argument"
trick to make them agree. You will meet it.

**A sub-matrix view.** Take the top-left 2x2 of a 4x6 matrix. The elements are
not contiguous — but they are regularly spaced:

```
full 4x6:                    the 2x2 view:
 0  1  2  3  4  5             0  1
 6  7  8  9 10 11             6  7
12 13 ...
```

Row stride is still 6 (the width of the *underlying* array), even though the
view is only 2 wide. So:

```
index = row * 6 + col        with row, col in [0,2)
```

**This is exactly what a tile is**, and it is why Chapter 06's tiled kernels
carry the original width around rather than the tile width. A stride that no
longer equals the logical width is the signature of a view into something
larger.

**3D and beyond.** A tensor `[D0, D1, D2]` in row-major, with the last
dimension fastest:

```
index = i0 * (D1 * D2) + i1 * D2 + i2

strides: (D1*D2, D2, 1)
```

And a transformer activation `[batch, heads, seq, dim]`:

```
index = b*(H*S*D) + h*(S*D) + s*D + d
```

That expression appears, with those exact names, in Chapter 19. You will have
seen it here first.

**The general rule for any number of dimensions:** the stride of a dimension is
the product of all the dimension sizes *after* it. The last dimension always
has stride 1 in row-major, which is another way of saying the last dimension is
the contiguous one.

---

## 4. Why strides decide performance

Everything so far has been about correctness. Here is where it becomes about
speed, and it is the bridge to Chapter 03.

From P0: memory moves in cache lines. Consecutive addresses are nearly free;
scattered addresses are expensive.

So consider summing a matrix two ways:

```cpp
// A: row by row
for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++)
        sum += m[r * W + c];

// B: column by column
for (int c = 0; c < W; c++)
    for (int r = 0; r < H; r++)
        sum += m[r * W + c];
```

Identical arithmetic. Identical answer. Identical number of memory accesses.

**A is typically 5 to 10 times faster than B on a small matrix, and 50 times
or more on a large one.** (On an M3 Max at 4096x4096, the exercise measures
64x. You will run it yourself in `p2-indexing`.)

In A, the inner loop increments `c`, so the index moves by 1 each step —
consecutive addresses, full use of every cache line. In B, the inner loop
increments `r`, so the index moves by `W` each step — a jump of `W * 4` bytes,
a brand-new cache line every single access, and 60 of every 64 bytes fetched
thrown away.

**The rule, which is worth memorising in this form:**

> Put the loop over the contiguous dimension innermost. The fastest-varying
> index should be the one with stride 1.

Chapter 03 says the same thing about threads instead of loop iterations:
consecutive *threads* should touch consecutive addresses. It is this chapter's
rule with 32 lanes doing it simultaneously, and if you understand it here you
will understand it there.

---

## 5. Transpose: the exercise that teaches indexing

Transposing swaps rows and columns: `out[c][r] = in[r][c]`. In flat form, with
input `H x W` and output `W x H`:

```cpp
for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++)
        out[c * H + r] = in[r * W + c];
```

Stare at the two index expressions. The input uses `W` (its width); the output
uses `H` (its width, which is the input's height). **Mixing those up is the
single most common indexing bug**, and it produces output that looks plausible
— especially if you test on a square matrix, where `H == W` and the bug is
invisible.

This is why `06-transpose` uses non-square, random matrices. A symmetric or
square test case hides exactly the bug you are most likely to have.

Notice also: one of these two accesses must be strided. Reading is contiguous
and writing jumps by `H`, or you swap the loops and the reverse happens. **You
cannot make both contiguous**, which is the entire reason Chapter 06 exists and
why tiling through shared memory is the answer.

---

## 6. Tiles and blocking

A tile is a rectangular sub-region processed as a unit. Walking a matrix in
`TxT` tiles:

```cpp
for (int tr = 0; tr < H; tr += T)            // tile row
    for (int tc = 0; tc < W; tc += T)        // tile col
        for (int r = tr; r < tr + T; r++)    // within the tile
            for (int c = tc; c < tc + T; c++)
                process(m[r * W + c]);
```

Four nested loops, and the inner two index into the *original* array with its
original stride `W`. The tile does not get its own array; it is a view.

Why do this? Because a tile is small enough to stay in cache. You load it once,
use every element of it several times, and move on — instead of streaming the
whole matrix past the processor repeatedly. The arithmetic is unchanged; the
*order* is chosen so that reuse happens while the data is still close by.

That is the entire idea of Chapter 06, and of every fast matrix multiply ever
written. You are looking at it now, with four `for` loops and no GPU in sight.

**The bounds problem.** If `H` is not a multiple of `T`, the last tile hangs
off the edge:

```cpp
for (int r = tr; r < tr + T && r < H; r++)   // clamp
```

Forgetting this reads past the end of the array — P2's first classic bug. In
CUDA it becomes `if (i < n)`, the bounds guard in every kernel you will write.
Same bug, same fix, bigger blast radius.

---

## 7. Common failure modes

Symptoms, so you can recognise them.

**Output is the transpose of what you wanted.** You swapped `row` and `col` in
one of the two index expressions.

**Works on square inputs, breaks on rectangular.** You used the wrong dimension
as the stride — `H` where you needed `W`. Always test non-square.

**First row correct, everything after it wrong.** Your row stride is wrong.
Check: does the index jump by the *underlying* width, or by the logical width
of your view?

**Last few elements wrong or a crash.** Missing bounds clamp on the final tile.

**Correct but inexplicably slow.** Your innermost loop is striding. Section 4.

**Off by exactly one row or column.** Someone wrote `<=` where they meant `<`,
or indexed from 1.

---

## 8. Exercises

Paper first for all of these. The arithmetic is the exercise; typing it is not.

### P3.1 - Flatten and unflatten

A matrix is 5 rows by 7 columns, row-major.

(a) Flat index of (row 3, col 4)?
(b) Coordinates of flat index 18?
(c) Flat index of the last element?
(d) How many elements total?
(e) Moving from (2,3) to (3,3): how much does the flat index change?
(f) Moving from (2,3) to (2,4): how much?
(g) Which of (e) and (f) is the stride-1 direction?

### P3.2 - Both orders

Same 5x7 matrix, now stored column-major.

(a) Flat index of (row 3, col 4)?
(b) Row and column strides?
(c) Element (0,0) is at flat 0 in both layouts. Which other elements are at the
    same flat index in both?
(d) Give the general condition for an element to be in the same place under
    both layouts.

### P3.3 - 3D

A tensor `[2, 3, 4]`, row-major.

(a) Strides of each dimension?
(b) Flat index of `[1, 2, 3]`?
(c) Coordinates of flat index 17?
(d) Total elements?
(e) Now `[batch=4, heads=8, seq=128, dim=64]`. Flat index of
    `[2, 3, 100, 20]`? (Take your time; this is the Chapter 19 expression.)
(f) In (e), which dimension is contiguous? If you wanted to loop over `seq`
    fastest, would that be efficient?

### P3.4 - Views

A 10x10 matrix. You want the 3x3 sub-matrix whose top-left corner is (4, 5).

(a) Flat index of the view's (0,0)?
(b) Flat index of the view's (1,1)?
(c) What is the row stride for walking the view?
(d) Write the general expression for view element (i, j).
(e) Is the view contiguous in memory? What would make it contiguous?

### P3.5 - Predict, then measure

(a) Write both loops from section 4 over a 4096x4096 float matrix.
(b) **Predict the ratio before running.** Write the number down.
(c) Time both. (`std::chrono::high_resolution_clock`, or just run each 10 times
    and use a stopwatch — precision does not matter when the effect is this
    large.)
(d) What ratio did you get? How far off was your prediction?
(e) Now try 64x64 instead. What happened to the ratio, and why? (Hint: how big
    is a 64x64 float matrix, and how big is your L1 cache?)
(f) Find, by bisection, the matrix size where the two loops stop differing.
    What does that size correspond to?

This exercise is the whole course in miniature: predict a number, measure it,
be wrong, find out why. Do not skip the prediction step.

### P3.6 - Transpose without looking back

(a) From memory, write the flat-index transpose from section 5.
(b) Test it on 3x5. Then 5x3. Then 4x4.
(c) Deliberately swap `H` and `W` in the output index. Which of your three
    tests still passes? Now you know why the square test is worthless.
(d) Which of the two accesses is strided in your version? Rewrite it so the
    other one is instead. Time both.
(e) Can you make both contiguous? Argue why not — carefully. That argument is
    Chapter 06's opening page.

---

## What you should now believe

- `index = row * width + col`, and `row = index / width`, `col = index % width`.
  Both directions, without thinking.
- A stride is how far the flat index moves when a coordinate moves by one. The
  last dimension of a row-major array has stride 1.
- The stride is the width of the *underlying* array, not of your view. That
  distinction is what makes tiles work.
- Loop order decides performance: the innermost loop must run along the stride-1
  dimension. Same arithmetic, an order of magnitude or two of difference.
- Transpose cannot make both the read and the write contiguous, which is why
  Chapter 06 exists.
- A tile is a view chosen so that reuse happens while the data is still in fast
  memory.
- Always test on non-square, non-power-of-two sizes. Square hides your bugs.

Next: [P4 - How numbers actually work](p4-numbers.md).
