# p0-hello

**Chapters:** [P0](../../p0-what-a-computer-is.md), [P1](../../p1-c-essentials.md)

## Task

Prove your compiler works, then fill in the byte arithmetic and the ceiling-
division idiom. Twenty minutes.

## Run

```bash
make run
```

You should see a list of `[pass]` and `[FAIL]` lines and a verdict. Everything
fails on the first run; that is correct.

## Pass bar

All checks pass.

## Notes

- Answer the `sizeof` questions **from memory first**, then confirm. You will
  be estimating memory traffic in your head for the rest of the course, and
  reaching for `sizeof` every time is not the same skill.
- `ceil_div` has a one-line implementation using only integer arithmetic. It
  must be exact at `a = 0` and when `b` divides `a` evenly — those are the two
  cases the obvious wrong answers get wrong.
- The last check is the single most common CUDA bug in existence, in CPU form.

## After it passes

1. Print the sizes with `sizeof` and confirm your memorised answers. Any
   surprises? (`long` differs between platforms; `int` is 4 bytes essentially
   everywhere you will care about.)
2. Deliberately break the build three ways — remove a semicolon, misspell
   `printf`, delete the `#include` — and read each error. P6.1 asks for this
   formally; doing it now costs two minutes.
3. Run `make -n run` and read the actual compiler command.
4. Write the table from P1.4: `n`, bytes, megabytes, for `n` up to 1e9. You
   will want it in P8.

<details>
<summary>Hints (questions only)</summary>

- `ceil_div`: what do you have to add to `a` before dividing so that any
  non-zero remainder pushes the quotient up by one? What is the largest
  remainder possible?
- Why does adding `b` instead of `b - 1` break the case where `b` divides `a`?
- For the last check: `1000 / 256` is what? Times 256 is what? So how many are
  left over?

</details>
