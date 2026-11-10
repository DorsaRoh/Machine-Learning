# Part -1 - Prerequisites

The main course starts by assuming you can read a pointer, flatten a 2D array
into 1D, and estimate how many bytes a loop touches. This part teaches those
things from nothing.

**Everything here runs on your laptop.** No GPU, no CUDA, no rented instance.
A plain C++ compiler, which macOS and Linux already have. That is deliberate:
you should not be paying for an A100 while you are still learning what a
pointer is.

---

## Should you skip this?

Answer these nine questions on paper. No compiler, no searching.

1. What is the difference between `int x = 5;` and `int* p = &x;`? What does
   `*p` mean, and what does `p` mean?
2. `float a[100];` — what is the type of `a + 7`, and what address does it hold
   if `a` is at address 1000?
3. A matrix is 4 rows by 6 columns, stored row-major in a flat array. Which
   flat index holds the element at row 2, column 3?
4. Same matrix. I move from row 2 column 3 to row 3 column 3. How far does the
   flat index move? From column 3 to column 4?
5. `0.1f + 0.2f == 0.3f` — true or false, and why?
6. What does `const float* p` stop you from doing? What about `float* const p`?
7. What is the difference between memory on the stack and memory on the heap?
   Which one survives past the end of the function that created it?
8. A loop reads 1 million floats and writes 1 million floats. How many bytes
   does it move?
9. If that loop takes 2 milliseconds, how many GB/s did it achieve?

**Got all nine, confidently?** Skip to
[Chapter 00](../chapters/00-why-gpus-exist.md). You are ready.

**Shaky on three or more?** Work through this part. It is six to ten hours of
real effort and it will save you weeks of confusion later, because every one of
those nine questions is something the main course uses on nearly every page
without explaining.

**Question 3 and 4 in particular.** If those were hard, do not skip this part
under any circumstances. Flat indexing of multi-dimensional data is not a
prerequisite for kernel programming, it *is* kernel programming. Chapter 03's
coalescing, Chapter 06's tiling and Chapter 12's register blocking are all the
same question asked with higher stakes.

---

## The chapters

| # | Chapter | What you walk away with |
|---|---|---|
| P0 | [What a computer actually is](p0-what-a-computer-is.md) | Bits, bytes, addresses, memory as one enormous array, why cache exists |
| P1 | [C essentials](p1-c-essentials.md) | Compiling and running, types, variables, functions, loops, `printf` |
| P2 | [Pointers and memory](p2-pointers-and-memory.md) | Addresses, dereferencing, arrays, stack vs heap, `const`, the mistakes everyone makes |
| P3 | [Arrays, indexing and strides](p3-arrays-and-indexing.md) | Flattening 2D and 3D, row-major, strides, tiles, bounds. **The most important chapter here** |
| P4 | [How numbers actually work](p4-numbers.md) | Binary, integer overflow, floating point, precision, why `a+b+c != c+b+a` |
| P5 | [Just enough C++](p5-cpp-for-cuda.md) | Headers, structs, references, `std::vector`, templates, lambdas — only the parts CUDA uses |
| P6 | [The toolchain](p6-toolchain.md) | Compiler, linker, `make`, flags, sanitizers, reading an error message without panic |
| P7 | [The math you need](p7-the-math.md) | Vectors, dot products, matrix multiply, reductions, softmax, normalisation |
| P8 | [Performance arithmetic](p8-performance-arithmetic.md) | Bytes, FLOPs, GB/s, prefixes, estimating a loop's cost before running it |

Read in order. P0 through P3 are the backbone; P4 through P8 each stand alone
if you need to come back to one.

---

## The exercises

Six of them. Do each one after the chapter it belongs to — the exercise
numbers do not line up with the chapter numbers, because some chapters are
reading and some are typing.

| Exercise | Do it after | Covers | Checks | Time |
|---|---|---|---|---|
| [`p0-hello`](exercises/p0-hello) | P0, P1 | compile, run, print, byte arithmetic, `ceil_div` | 15 | 20 min |
| [`p1-pointers`](exercises/p1-pointers) | P2 | pointers, arrays, output parameters, pointer-to-pointer | 20 | 45 min |
| [`p2-indexing`](exercises/p2-indexing) | P3 | flattening, strides, 4D tensors, transpose, tiles, loop order | 39 | 90 min |
| [`p3-floats`](exercises/p3-floats) | P4 | precision limits, summation order, cancellation, NaN | 35 | 60 min |
| [`p4-math`](exercises/p4-math) | P7 | dot, matmul, softmax, LayerNorm, and attention | 24 | 120 min |
| [`p5-estimate`](exercises/p5-estimate) | P8 | bytes, FLOPs, intensity, predict-then-measure | 28 | 60 min |

If you are short on time: **`p2-indexing` and `p5-estimate` are the two that
matter most.** The first is the skill every kernel is made of; the second is
the habit that tells you whether a kernel is finished.

Same rule as the rest of the course: **you are not given answers.** Exercises
are skeletons with `// TODO`, hints are questions, and there are no solution
files. The checks tell you *what* is wrong, never *how* to fix it.

```bash
cd prerequisites/exercises/p0-hello
make run
```

If something crashes or gives nonsense, before anything else:

```bash
make debug
```

That rebuilds with the address and undefined-behaviour sanitizers, which will
tell you the exact line where you read off the end of an array. It is the
laptop-sized version of `compute-sanitizer`, which you will meet again in
Chapter 18.

---

## How these map onto the real course

Nothing here is busywork. Each chapter exists because a specific later chapter
would otherwise be impossible:

| Prerequisite | Why |
|---|---|
| P2 pointers | Every CUDA kernel signature is pointers. `__global__ void f(const float* in, float* out)` |
| P3 indexing | Chapter 03 coalescing, Chapter 06 tiling, Chapter 11 GEMM are all flat-index arithmetic |
| P4 floating point | Chapter 15 numerics, and the reason reductions give different answers on a GPU |
| P5 C++ | `bench.cuh` uses templates, lambdas and `std::vector`; you need to read it |
| P6 toolchain | `make run`, `make sanitize`, and understanding compiler errors |
| P7 math | Chapters 07, 10, 11, 19 are reductions, softmax and matrix multiply |
| P8 estimation | Chapter 01's roofline is this chapter with the GPU's numbers substituted in |

Start with [P0](p0-what-a-computer-is.md).
