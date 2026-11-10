# P1 - C essentials

> Enough C to read and write a kernel. Not a C course: there is a great deal of
> C you will never need here, and this chapter leaves all of it out.

---

## 1. The smallest complete program

Put this in a file called `hello.cpp`:

```cpp
#include <cstdio>

int main() {
    std::printf("hello\n");
    return 0;
}
```

Compile and run it:

```bash
c++ -std=c++17 -O2 -Wall hello.cpp -o hello
./hello
```

Line by line, because every piece recurs forever:

- `#include <cstdio>` — paste in the declarations for the printing functions.
  Without it the compiler has never heard of `printf`.
- `int main()` — where execution starts. Every program has exactly one.
- `std::printf(...)` — print. `\n` is a newline character.
- `return 0;` — tell the operating system "finished, no errors". Non-zero means
  failure, which is how `make run` knows whether your exercise passed.
- Semicolons end statements. Braces `{ }` group them.

`c++` is the compiler. `-std=c++17` picks the language version, `-O2` turns on
optimisation, `-Wall` turns on warnings. `-o hello` names the output.
`./hello` runs it.

---

## 2. Variables and types

```cpp
int   count = 10;         // whole number, 4 bytes
float x     = 3.14f;      // decimal, 4 bytes. The f matters; see below
double y    = 3.14;       // decimal, 8 bytes
char  c     = 'A';        // one byte
bool  flag  = true;       // true or false
```

Every variable has a **type**, fixed when you declare it, and the type decides
how many bytes it occupies and how the bits are interpreted (P0 section 1).

**The `f` suffix.** `3.14` is a `double`; `3.14f` is a `float`. Write
`float x = 3.14;` and the compiler converts a double to a float at every use,
which is slower and occasionally surprising. In GPU code this matters more than
you would think — a stray `double` constant can drag a whole expression into
double precision, and GPUs are often 32x slower at double than at float. Get in
the habit now: **float literals get an `f`.**

**Integer division truncates.**

```cpp
int a = 7 / 2;        // 3, not 3.5
float b = 7 / 2;      // 3.0f. The division happened in ints BEFORE the assignment
float c = 7.0f / 2;   // 3.5f
```

This bites everyone at least once. When you compute an average or a midpoint
and get a suspiciously round number, check for integer division.

**Integer division is also how you compute grid sizes**, so you will use the
truncation deliberately:

```cpp
int blocks = (n + 255) / 256;   // "round up": how many blocks of 256 cover n?
```

Work out why that rounds up rather than down — it appears in literally every
CUDA launch you will write.

---

## 3. Control flow

```cpp
if (x > 10) {
    std::printf("big\n");
} else if (x > 5) {
    std::printf("medium\n");
} else {
    std::printf("small\n");
}

for (int i = 0; i < 10; i++) {
    std::printf("%d\n", i);        // 0 through 9
}

int i = 0;
while (i < 10) {
    i++;
}
```

The `for` loop, spelled out: initialise `i = 0`; before each pass check
`i < 10`; after each pass do `i++`. Loops in C conventionally run `0` to `n-1`,
not `1` to `n`, because arrays are indexed from zero.

**Comparison is `==`, assignment is `=`.** `if (x = 5)` assigns 5 to `x` and is
always true. `-Wall` warns about this, which is one reason it is on.

Operators you will see constantly:

```cpp
i++        // add 1
i += 4     // add 4
n % 4      // remainder: 7 % 4 is 3
n / 4      // integer division: 7 / 4 is 1
x && y     // and          x || y    // or         !x    // not
```

`%` and `/` together are how you decompose a flat index into coordinates, which
is the whole of P3. `7 / 4 = 1` and `7 % 4 = 3` says "7 is at row 1, column 3
in a 4-wide grid". Hold that thought.

---

## 4. Printing, so you can see what is happening

`printf` takes a format string with placeholders:

```cpp
int n = 42;
float x = 3.5f;
double d = 1.0 / 3.0;
const char* name = "kernel";

std::printf("integer: %d\n", n);
std::printf("float:   %f\n", x);          // 3.500000
std::printf("neater:  %.3f\n", x);        // 3.500
std::printf("science: %.3e\n", d);        // 3.333e-01
std::printf("either:  %g\n", d);          // picks a sensible format
std::printf("string:  %s\n", name);
std::printf("hex:     %x\n", n);          // 2a
std::printf("pointer: %p\n", (void*)&n);  // an address
std::printf("size_t:  %zu\n", sizeof(n)); // 4
std::printf("two: %d and %.2f\n", n, x);  // placeholders fill in order
```

| Placeholder | For |
|---|---|
| `%d` | `int` |
| `%lld` | `long long` |
| `%zu` | `size_t` (a size or count; always unsigned) |
| `%f` `%g` `%e` | `float` or `double` |
| `%s` | a C string |
| `%p` | a pointer (cast it to `void*`) |
| `%x` | an integer in hexadecimal |

**Mismatching the placeholder and the argument is undefined behaviour** — it
prints garbage or crashes rather than erroring. `%d` with a float is the common
one. `-Wall` catches most of these; read its warnings.

`printf` is not elegant, and C++ has `std::cout`. This course uses `printf`
because **it works inside CUDA kernels** and `cout` does not, so you may as
well be fluent in the one that will still be available in Chapter 02.

---

## 5. Functions

```cpp
// return type, name, parameters
float add(float a, float b) {
    return a + b;
}

// no return value
void greet(const char* who) {
    std::printf("hello %s\n", who);
}

int main() {
    float s = add(2.0f, 3.0f);
    greet("world");
    return 0;
}
```

A function must be **declared before it is used**. Either define it above
`main`, or write a declaration first:

```cpp
float add(float a, float b);   // declaration: "this exists, here is its shape"
```

That split — declaration separate from definition — is what header files are
for (P5).

**Arguments are copied.** This is the single most important thing about C
functions:

```cpp
void try_to_change(int x) {
    x = 99;                 // changes the copy
}

int main() {
    int a = 5;
    try_to_change(a);
    std::printf("%d\n", a); // 5. Not 99.
}
```

`try_to_change` received a *copy* of `a`. Modifying the copy does nothing to
the original. To actually change the caller's variable you must pass its
address — which is a pointer, which is P2, and which is why every CUDA kernel
signature is full of them.

---

## 6. Common mistakes, and what they look like

**Missing semicolon.** The error usually points at the *next* line. If a line
looks fine, check the one above it.

**Using a variable before giving it a value.**

```cpp
int sum;                              // contains whatever bytes were there
for (int i = 0; i < n; i++) sum += a[i];   // garbage
```

`int sum = 0;` — always initialise. `-Wall` warns; `make debug` catches it at
runtime.

**Integer division where you wanted real division.** See section 2.

**`=` instead of `==` in a condition.** See section 3.

**Declaring in a loop and using it outside.**

```cpp
for (int i = 0; i < 10; i++) { }
std::printf("%d\n", i);   // error: i does not exist here
```

A variable exists only inside the `{ }` braces where it was declared. That is
its **scope**.

---

## 7. Exercises

Do these in a scratch file, not in the exercise directory.

### P1.1 - Get a compiler working

(a) Write, compile and run `hello.cpp`.
(b) Delete a semicolon. Compile. Read the error. Note which line it blames.
(c) Misspell `printf`. Compile. How is that error different from (b)?
(d) Remove the `#include`. Compile. Again — how is this error different?
(e) You have now seen the three families of error — syntax, unknown name,
    missing declaration. Being able to tell them apart on sight is worth more
    than it sounds.

### P1.2 - Integer division

Predict each, then check:

```cpp
std::printf("%d\n", 7 / 2);
std::printf("%d\n", -7 / 2);
std::printf("%d\n", 7 % 3);
std::printf("%f\n", 7 / 2);          // careful
std::printf("%f\n", 7.0f / 2);
std::printf("%d\n", (10 + 3) / 4);
std::printf("%d\n", (10 + 4 - 1) / 4);
```

(f) The last two are "round up" idioms. Which one is right, and why does the
    other give the wrong answer when `n` divides evenly?

### P1.3 - The ceiling-division idiom

(a) Write `int ceil_div(int a, int b)` returning `a / b` rounded up.
(b) Check it for `a` = 0 through 10 with `b` = 4.
(c) How many blocks of 256 threads cover 1000 elements? 1024? 1025?
(d) If you launched `n / 256` blocks instead, how many of the 1000 elements
    would be missed? This is the single most common CUDA bug.

### P1.4 - A loop that prints a table

Write a program that prints, for `n` = 1, 10, 100, ... 1e9:
`n`, the bytes `n` floats occupy, and that in megabytes. Use `%12d` and `%10.3f`
to line the columns up. You will want this table in P8.

---

## What you should now believe

- A program is `main`, some functions, and a compiler command.
- Types fix the size and interpretation of the bits. `float` literals take `f`.
- Integer division truncates, which is a bug in averages and a tool in grid
  sizing.
- `printf` placeholders must match their arguments, and `%d` with a float is
  not an error, just wrong.
- Function arguments are copied, so a function cannot change its caller's
  variable — unless you hand it an address.

Next: [P2 - Pointers and memory](p2-pointers-and-memory.md), which is the
chapter that makes the rest of the course possible.
