# p3-floats

**Chapter:** [P4 - Numbers](../../p4-numbers.md)

## Task

Find the edges of `float` by hand: where it stops counting, how much a sum
loses, how a textbook formula produces a *negative variance*, and why softmax
needs a trick. Sixty minutes.

## Run

```bash
make run
```

## Pass bar

All 35 checks pass.

## Notes

- **Nothing here is looked up.** `first_unrepresentable_integer` and
  `float_epsilon` are found with loops. Typing `16777217` because you read it
  in P4 defeats the exercise; writing the four-line loop that discovers it
  means you will never again be surprised by a model whose loss stops
  improving.
- **Part 4 is the centrepiece.** Three ways to add up twenty million ones.
  Sequential float gets 16,777,216. Tree-structured float — same precision,
  same data, same number of additions — gets 20,000,000 exactly. That gap is
  why Chapter 07's reduction is a tree, and why a GPU result that differs from
  your CPU reference in the sixth digit is not necessarily a bug.
- **Part 5 will produce a negative variance.** Variance is a sum of squares; it
  cannot be negative. The reference run of this exercise gets `-634237760`.
  When you see that number, you have understood catastrophic cancellation
  better than any explanation could manage.
- **Part 6 asks you to write a broken softmax on purpose.** Run it, watch it
  print `nan nan nan`, and then notice that your stable version gets the right
  answer on exactly the same input. That one subtraction is load-bearing for
  every transformer ever trained.

## After it passes

1. Change `sum_tree_float`'s cutoff — the size below which you stop recursing
   and sum sequentially. Try 2, 64, 1024, 100000. At what point does the answer
   start degrading? That threshold is a precision-versus-overhead tradeoff and
   you have just found yours.
2. Run part 4 again with the array filled with `0.1f` instead of `1.0f`. What
   is the true answer? How do you *know*? (This is harder than it sounds and is
   the real lesson of P4.)
3. Make the `is_nan` check fail: compile with `-ffast-math` and re-run. Which
   checks break? `-ffast-math` tells the compiler it may assume no NaNs and may
   reassociate floating point — which is roughly what `--use_fast_math` does to
   a CUDA kernel in Chapter 15.
4. Print the raw bits of a few floats (`memcpy` into a `uint32_t`, then print
   in hex). Look at `1.0f`, `2.0f`, `0.0f`, `-0.0f`, `inf`, `nan`. Can you see
   the sign/exponent/mantissa split from P4 section 3 in the hex?
5. Implement `fp16`-style rounding by hand: round a float to 10 mantissa bits
   and back. How large does a number get before the gap exceeds 1? That number
   is why Chapter 21 keeps a separate scale factor.

<details>
<summary>Hints (questions only)</summary>

- `first_unrepresentable_integer`: if `n` and `n+1` land on the same float,
  which of the two is the one that cannot be represented?
- `float_epsilon`: start at 1 and halve. Do you want the last value for which
  `1 + e != 1`, or the first for which `1 + e == 1`? Those differ by a factor
  of two and only one of them is epsilon.
- `where_counting_stalls`: what does the loop condition look like when nothing
  is changing any more? And how do you avoid waiting for 17 million iterations?
- `sum_tree_float`: what is the base case? What happens when `n` is odd?
- `variance_naive_float`: *every* intermediate must be a `float`, including the
  mean and the two sums. If you accumulate in double by accident, the check
  will tell you it is not wrong enough.
- `softmax_naive`: write the formula exactly as it appears in a textbook, with
  no shift at all. That is the whole task.
- `is_nan`: IEEE 754 defines exactly one value for which `x == x` is false.

</details>
