# p1-pointers

**Chapter:** [P2 - Pointers and memory](../../p2-pointers-and-memory.md)

## Task

Nine small functions, all shaped like CUDA kernels: handed addresses and a
length, writing their answers through pointers. Forty-five minutes.

## Run

```bash
make run          # optimised build
make debug        # AddressSanitizer + UBSan, for when it crashes
```

## Pass bar

All 20 checks pass.

## Notes

- Nothing here returns an array, because in C nothing can. That restriction is
  the reason every kernel signature in the main course ends with an output
  pointer and a length.
- `find_largest` is the hard one. It takes `const float**` — a pointer to a
  pointer — because it has to change a pointer variable that belongs to the
  caller. Read P2 section 5 again if that sentence does not land; it is the
  same shape as `cudaMalloc(&d_a, bytes)`, which you will type a hundred times.
- `sum` accumulates 20 million ones. If your answer comes back as exactly
  `16777216`, you have just rediscovered 2^24 by accident, which is the best
  way to learn it.
- Two checks test edge cases rather than the happy path: `n = 0`, and
  `swap_ints(&z, &z)` where both arguments are the same address. A kernel that
  does the right thing on 1,000,000 elements and the wrong thing on 0 is a
  kernel that will fail in production at 3am.

## After it passes

1. Print the pointers themselves with `%p` and look at the actual addresses.
   What is the difference between `&ramp[0]` and `&ramp[1]`, in bytes? Between
   `&int_array[0]` and `&int_array[1]`?
2. Change one `const float*` to `float*` in a function that only reads. Does it
   still compile? Now do the reverse — add `const` to a parameter you write
   through. Read the error carefully; it is the one that protects you.
3. Do P2.5 (break it on purpose): read off the end of an array, free something
   twice, dereference null. Run each under `make debug` and collect the three
   error messages. These are the messages `compute-sanitizer` will give you in
   Chapter 18, with GPU nouns substituted.
4. Rewrite `add_arrays` using only pointer arithmetic — no `[]` anywhere.
   Convince yourself it is the same program.

<details>
<summary>Hints (questions only)</summary>

- `swap_ints`: why does the aliased case `swap_ints(&z, &z)` work with a
  temporary, and what would break without one?
- `min_max`: what should the initial value of your running minimum be? Why is
  `0` wrong, and why is `data[0]` right? What does that imply about `n = 0`?
- `find_largest`: you need to track a *location*, not a value. What type is
  your running-best variable? When you finally write the answer out, how many
  stars are involved?
- `element_at`: if `base + 1` moves 4 bytes, what moves `k` elements?

</details>
