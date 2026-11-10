# P5 - Just enough C++

> C++ is enormous. CUDA uses a small, specific corner of it. This chapter is
> that corner and nothing else — chosen by going through `common/bench.cuh` and
> the exercise harnesses and listing what actually appears.

---

## 1. Headers and why code is split in two

A **header** (`.h`, `.hpp`, `.cuh`) declares what exists. A **source file**
(`.cpp`, `.cu`) defines what it does.

```cpp
// math_utils.h
#ifndef MATH_UTILS_H        // include guard: stops double-inclusion
#define MATH_UTILS_H

float add(float a, float b);     // declaration only

#endif
```

```cpp
// math_utils.cpp
#include "math_utils.h"

float add(float a, float b) { return a + b; }   // definition
```

`#include "file.h"` literally pastes the file in at that point. The include
guard stops you pasting it twice, which would be a redefinition error.

`#include <cstdio>` with angle brackets searches the system directories;
`#include "mine.h"` with quotes searches your directory first. That is the only
difference.

In this course you will see exercises `#include "kernel.cu"` directly into
`main.cu`. That is unusual for production C++ and deliberate here: it keeps
each exercise a single compilation unit so there is no build system to fight.

**`inline`** lets you put a function definition in a header without the linker
complaining about duplicates. `bench.cuh` is full of `inline` functions for
this reason.

---

## 2. Structs: grouping data

```cpp
struct DeviceInfo {
    int sm_count;
    int warp_size;
    double bandwidth_gbs;
};

DeviceInfo info;
info.sm_count = 108;
std::printf("%d\n", info.sm_count);
```

A `struct` is several values with names, stored consecutively. Through a
pointer you use `->` instead of `.`:

```cpp
DeviceInfo* p = &info;
p->sm_count = 108;        // same as (*p).sm_count
```

You can give members default values and initialise with braces:

```cpp
struct ErrorStats {
    double max_abs = 0;
    double max_rel = 0;
    size_t first_bad = size_t(-1);
};

ErrorStats s;              // uses the defaults
ErrorStats t{1.0, 2.0};    // sets the first two
```

`bench::ErrorStats` is exactly this, and the harnesses rely on the defaults.

C++ `class` is the same thing with members private by default. `bench::Gate` is
a class: data hidden, a few functions exposed. You do not need to write classes
in this course, only read them.

---

## 3. References: pointers with less punctuation

```cpp
void scale(float& x, float k) { x *= k; }   // note the &

float v = 2.0f;
scale(v, 3.0f);     // v is now 6.0f. No & at the call site.
```

A **reference** is another name for an existing variable. It behaves like a
pointer that is always valid and automatically dereferenced.

| | Pointer | Reference |
|---|---|---|
| Declare | `int* p = &x;` | `int& r = x;` |
| Use | `*p = 5;` | `r = 5;` |
| Can be null | yes | no |
| Can be reseated | yes | no |
| Call site | `f(&x)` | `f(x)` |

The form you will see constantly is **const reference**, used to pass something
large without copying it:

```cpp
void print(const std::vector<float>& v);    // no copy, cannot modify
```

Without the `&` that would copy an entire vector on every call.

**References are host-side C++.** Device code uses raw pointers, because a
kernel's parameters are addresses in GPU memory. So you will read references in
`bench.cuh` and write pointers in `kernel.cu`.

---

## 4. `std::vector`: an array that manages itself

```cpp
#include <vector>

std::vector<float> v(1000);          // 1000 floats, all zero
std::vector<float> w(1000, 3.5f);    // 1000 floats, all 3.5
std::vector<float> u = {1, 2, 3};    // from a list

v[0] = 1.0f;                          // index like an array
v.size();                             // how many elements
v.push_back(4.0f);                    // append, growing if needed
v.data();                             // a float* to the underlying memory
```

A `vector` is a heap array (P2 section 5) that knows its own length and frees
itself when it goes out of scope. No `new`, no `delete`, no leaks.

**`.data()` is the bridge to CUDA.** Every exercise does this:

```cpp
std::vector<float> h_a(n);                                 // host data
cudaMemcpy(d_a, h_a.data(), n * sizeof(float), ...);       // to the GPU
```

`cudaMemcpy` wants a raw pointer; `.data()` hands over the address of the
vector's first element. The vector still owns and will still free the memory.

The `<float>` is a **template argument** — see the next section.

Two things to keep in mind:

- `v[i]` does **not** bounds-check. `v.at(i)` does, and throws. The exercises
  use `[]` because that is what the main course uses; `make debug` is your
  bounds check.
- A vector lives on the host. There is no `std::vector` in a CUDA kernel.

---

## 5. Templates: code with a type as a parameter

```cpp
template <typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

maximum(3, 5);        // T is int
maximum(3.5f, 2.1f);  // T is float
```

The compiler generates a separate copy of the function for each type you use
it with. That is why templates live in headers: the definition must be visible
wherever it is instantiated.

A template parameter can also be a **value**, which is the form that matters
for CUDA:

```cpp
template <int BLOCK_SIZE>
__global__ void reduce(float* data) {
    __shared__ float tile[BLOCK_SIZE];     // size known at compile time
    ...
}

reduce<256><<<grid, 256>>>(d_data);
```

`BLOCK_SIZE` is fixed when the code is compiled, so the compiler can size
arrays with it, fully unroll loops over it, and specialise the generated
machine code. `04-occupancy-sweep` uses `template <int ILP>` and
`14-pipelined-gemm` uses a stage count this way. You will mostly *use*
templates rather than write them, but you need to read the syntax without
stumbling.

---

## 6. Lambdas: a function written where it is used

```cpp
auto square = [](float x) { return x * x; };
std::printf("%f\n", square(3.0f));    // 9.0
```

`[]` is the **capture list** — which surrounding variables the lambda can see:

```cpp
int n = 10;
auto f = [n]()  { return n * 2; };    // captures a copy of n
auto g = [&n]() { return n * 2; };    // captures a reference to n
auto h = [&]()  { return n * 2; };    // captures everything by reference
```

You need this to read the timing code, which is in every harness:

```cpp
double ms = bench::time_ms([&] { my_kernel<<<grid, block>>>(d_a, d_b, n); });
```

That passes a chunk of code *as a value* so the timer can run it many times.
`[&]` lets the lambda see `d_a`, `d_b`, `n`, `grid` and `block` from the
enclosing function. You will write these in every exercise from Chapter 02 on.

**`auto`** asks the compiler to work out the type. Essential for lambdas (whose
types are unnameable) and convenient elsewhere. Do not overuse it; `auto x = 0`
being an `int` rather than the `float` you wanted is a real bug.

---

## 7. Namespaces

```cpp
namespace bench {
    double gbs(size_t bytes, double ms);
}

bench::gbs(1000, 2.0);      // qualified call
```

A namespace is a prefix that keeps names from colliding. `std::` is the
standard library's; `bench::` is this course's.

You will see `using namespace std;` in tutorials. Do not write it — it drags
thousands of names into scope and produces ambiguity errors that are miserable
to read. Type `std::` out.

---

## 8. The small things that appear constantly

**Macros.** Text substitution before compilation.

```cpp
#define GROUP 128                 // a named constant
#define CUDA_CHECK(expr) ...      // wraps every CUDA call, in every exercise
```

Prefer `constexpr int group = 128;` for constants in modern C++, but macros are
unavoidable in CUDA error-checking because they need `__FILE__` and `__LINE__`.

**Conditional compilation.**

```cpp
#ifdef HAVE_NCCL
    // only compiled when the flag is set
#endif
```

`25-allreduce` uses this so it still builds without NCCL.

**Casts.**

```cpp
float f = (float)i;              // C style, works
float g = static_cast<float>(i); // C++ style, preferred, easier to search for
size_t idx = size_t(row) * w;    // functional style, common in this course
```

**`sizeof`.** Bytes of a type or object. `sizeof(float)` is 4.

**`const` on functions and parameters.** P2 section 6. Mark inputs `const`.

**`nullptr`.** The null pointer. Use it rather than `NULL` or `0`.

---

## 9. What you will *not* need

So you do not go off and learn the wrong things: this course uses essentially
none of inheritance, virtual functions, operator overloading, exceptions, smart
pointers, move semantics, the STL algorithms beyond `std::sort`, or the
iostream library. If a C++ tutorial spends three chapters on class hierarchies,
that is a fine thing to know and it is not this.

---

## 10. Exercises

### P5.1 - Read the harness

Open `Kernels/common/bench.cuh` and find one example of each:

(a) A struct with default member values.
(b) An `inline` function, and an explanation of why it must be `inline`.
(c) A function template taking a lambda.
(d) A const reference parameter, and what it avoids copying.
(e) A class with private members.
(f) For each, write one sentence on what it does. You do not need to
    understand the CUDA parts yet — just the C++ shape.

### P5.2 - Vectors and raw pointers

(a) Make a `std::vector<float>` of 1000 elements, fill with `i * 0.5f`.
(b) Write `float sum(const float* data, int n)` taking a raw pointer.
(c) Call it with `.data()` and `.size()`.
(d) Why does `sum` take a pointer rather than the vector? (Think about what a
    CUDA kernel can accept.)
(e) Pass `v.data() + 500` and `500`. What did you just compute? Relate it to
    P2.4.

### P5.3 - Lambdas and timing

(a) Write `template <typename F> double time_it(F&& fn, int iters)` that runs
    `fn` `iters` times and returns the average milliseconds, using
    `std::chrono::high_resolution_clock`.
(b) Use it to time a lambda that sums a large vector.
(c) Capture the vector by reference. What happens if you capture by value
    instead — both to correctness and to the time?
(d) You have just written a simplified `bench::time_ms`. Open the real one and
    list three things it does that yours does not.

### P5.4 - A template you have to instantiate

(a) Write `template <int N> float sum_fixed(const float* a)` that sums exactly
    `N` elements.
(b) Call it as `sum_fixed<8>(data)`.
(c) Compile with `-O2 -S` and look at the generated assembly, or just time it
    against a normal loop over 8 elements with a runtime `n`.
(d) Why can the compiler do more with the template version? This is the entire
    reason CUDA kernels take block sizes as template parameters.

---

## What you should now believe

- Headers declare, source files define, and `#include` is literal text
  pasting.
- A struct groups named values; `.` through a value, `->` through a pointer.
- A reference is another name for a variable; `const T&` passes big things
  without copying. Host-side only.
- `std::vector` is a self-managing heap array, and `.data()` is how it hands a
  raw pointer to CUDA.
- Templates parameterise code by type *or by value*, and the value form is how
  CUDA kernels get compile-time block sizes.
- Lambdas with `[&]` are how you hand a block of code to a timing function.
- You need a small, specific corner of C++, and the rest can wait indefinitely.

Next: [P6 - The toolchain](p6-toolchain.md).
