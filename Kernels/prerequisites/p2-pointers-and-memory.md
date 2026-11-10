# P2 - Pointers and memory

> Pointers have a reputation for being hard. They are not hard; they are
> *unfamiliar*. There is exactly one idea here, and once it lands, every CUDA
> kernel signature you will ever read becomes obvious.

---

## 1. The one idea

From P0: **memory is one enormous numbered array of bytes, and a variable is a
name for an address.**

A **pointer is a variable whose value is an address.**

That is it. Everything else is notation.

```cpp
int x = 42;        // somewhere in memory, say address 1000, holds 42
int* p = &x;       // p is a variable holding the number 1000
```

Two operators do all the work:

| Operator | Reads as | Does |
|---|---|---|
| `&x` | "address of x" | gives you where `x` lives |
| `*p` | "the thing at p" | goes to that address and uses what is there |

They are opposites. `&` turns a thing into an address; `*` turns an address
back into a thing.

```cpp
int x = 42;
int* p = &x;

std::printf("%d\n", x);          // 42   the value
std::printf("%p\n", (void*)&x);  // 1000 the address of x
std::printf("%p\n", (void*)p);   // 1000 p holds that address
std::printf("%d\n", *p);         // 42   go to 1000, read what is there

*p = 99;                          // go to 1000, write 99 there
std::printf("%d\n", x);          // 99   because x IS address 1000
```

The last two lines are the point. `*p = 99` and `x = 99` do exactly the same
thing, because `p` holds `x`'s address. There is no copy. There is no link.
There is one piece of memory with two ways to reach it.

**Picture it:**

```
     name      address      contents
      x    ->   1000    ->    [ 42 ]
                                 ^
      p    ->   2000    ->    [ 1000 ]
```

`p` is an ordinary variable at its own address, and the thing it stores happens
to be a number that is also an address.

---

## 2. The `*` in a declaration means something different

This trips up everyone:

```cpp
int* p = &x;    // DECLARATION:  "p is a pointer to int"
*p = 99;        // STATEMENT:    "go to where p points, write 99"
```

Same symbol, two jobs. In a declaration it is part of the *type*. In an
expression it is the *dereference* operation.

A useful habit: read declarations **right to left**.

```cpp
int* p;              // p is a pointer to int
const float* q;      // q is a pointer to a const float
float* const r = &y; // r is a const pointer to a float
```

Where you put the `*` is cosmetic — `int* p`, `int *p` and `int * p` are the
same declaration. This course writes `int* p`, attaching it to the type,
because the type really is "pointer to int".

One genuine trap:

```cpp
int* a, b;    // a is a pointer. b is a plain int. Almost never what you meant.
int *a, *b;   // both are pointers
```

Declare one per line and the problem disappears.

---

## 3. Why pointers exist: changing the caller's variable

P1 section 5 left this hanging. Arguments are copied, so a function cannot
modify the caller's variable:

```cpp
void broken(int x)  { x = 99; }        // changes a copy

void works(int* x)  { *x = 99; }       // changes the original

int main() {
    int a = 5;
    broken(a);      std::printf("%d\n", a);   // 5
    works(&a);      std::printf("%d\n", a);   // 99
}
```

`works` receives the *address* of `a`. The address is itself copied — but a
copy of an address still points at the same memory. So `*x = 99` reaches out
and modifies the original.

**This is the reason every CUDA kernel looks the way it does:**

```cuda
__global__ void add(const float* a, const float* b, float* c, int n)
```

`a`, `b` and `c` are addresses of arrays living in GPU memory. The kernel
cannot *return* a million results, so instead it is handed the address of
where to put them. Once you see `float* c` as "here is where the output goes",
every signature in this course reads naturally.

---

## 4. Arrays are pointers in disguise

```cpp
float a[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
```

This reserves 5 consecutive floats — 20 bytes. If the first is at address 1000:

```
index:       0       1       2       3       4
address:   1000    1004    1008    1012    1016
value:     10.0    20.0    30.0    40.0    50.0
```

Each step of 1 in the index is a step of 4 bytes in the address, because a
`float` is 4 bytes.

**The array's name is the address of its first element.** `a` and `&a[0]` are
the same thing. So this works:

```cpp
float* p = a;              // no & needed; a is already an address
std::printf("%f\n", p[2]); // 30.0
std::printf("%f\n", a[2]); // 30.0, identical
```

And here is the definition that explains everything:

```
a[i]  means  *(a + i)
```

"Add `i` to the address, then look at what is there."

**Pointer arithmetic counts in elements, not bytes.** `a + 1` is not address
1001, it is address 1004 — one `float` further on. The compiler knows the type,
so it scales for you:

```cpp
float* p = a;
p + 1      // 1004   (4 bytes further)
p + 3      // 1012
*(p + 3)   // 40.0, same as p[3]
```

If `p` were a `double*`, `p + 1` would move 8 bytes. The arithmetic depends on
the pointed-to type, which is exactly why `void*` (pointer to nothing in
particular) cannot be dereferenced or incremented.

**Arrays decay to pointers when passed to functions**, and the length is lost:

```cpp
void process(float* data, int n) { ... }   // n is not optional

float a[5];
process(a, 5);                             // you must pass the length yourself
```

Inside `process`, `sizeof(data)` is the size of a pointer (8 bytes), not the
size of the array. **C does not know how long your arrays are.** Every
function that takes an array takes a length alongside it, and every CUDA kernel
in this course takes an `int n` for exactly this reason.

---

## 5. Stack and heap

Two places your data can live, and the difference decides whether it survives.

**The stack** is automatic. Local variables go here; they vanish when the
function returns.

```cpp
void f() {
    float a[100];      // 400 bytes on the stack
}                      // gone
```

Fast, automatic, and small — typically 1-8 MB for the whole program. Declaring
`float a[10000000];` as a local will crash with a stack overflow.

**The heap** is manual. You ask for memory, you get an address, and it stays
until you release it.

```cpp
float* a = new float[1000000];   // 4 MB, address returned
// ... use it ...
delete[] a;                       // release it
```

In C, and in a lot of CUDA host code, the same thing spelled differently:

```cpp
float* a = (float*)malloc(1000000 * sizeof(float));
free(a);
```

Big, long-lived, and yours to manage. **The GPU version is the same idea:**

```cuda
float* d_a;
cudaMalloc(&d_a, n * sizeof(float));   // allocate in GPU memory
cudaFree(d_a);
```

Note `&d_a`: `cudaMalloc` has to *change* your pointer variable, so it needs
the address of the pointer — a pointer to a pointer. If section 3 landed, this
is obvious rather than weird.

**`sizeof` is how you compute byte counts**, and you will do it constantly:

```cpp
n * sizeof(float)      // bytes for n floats. Never write n * 4 by hand.
```

---

## 6. `const`, and why it is not bureaucracy

```cpp
const float* p;        // cannot change *p. Can point p somewhere else.
float* const q = &y;   // can change *q. Cannot point q somewhere else.
```

Right-to-left again: `const float* p` is "p is a pointer to a const float".

The form you will see everywhere in this course is the first one:

```cuda
__global__ void add(const float* a, const float* b, float* c, int n)
```

It says: **`a` and `b` are inputs, `c` is the output.** That is useful
documentation for a reader — and more than documentation for the compiler,
which can optimise more aggressively when it knows a value will not change.
In CUDA, `const` combined with `__restrict__` (Chapter 03) lets the compiler
use a faster read-only cache path. So it is a performance annotation, not just
a safety one.

Get in the habit: **if a function does not write through a pointer, mark it
`const`.**

---

## 7. The four ways to get it wrong

These are the bugs. All four are things the compiler will cheerfully let you
do, and all four are caught by `make debug`.

**Out of bounds.** The classic.

```cpp
float a[5];
a[5] = 1.0f;     // indices are 0..4. This writes past the end.
```

Nothing stops you. You have overwritten whatever happened to live there, and
the symptom will appear somewhere unrelated, later. Chapter 02's bounds check
(`if (i < n)`) exists precisely because the GPU version of this mistake is
even easier to make.

**Uninitialised pointer.**

```cpp
int* p;          // holds a garbage address
*p = 5;          // writes 5 to a random place. Crash, if you are lucky.
```

Always initialise: `int* p = nullptr;` and check before dereferencing.

**Use after free.**

```cpp
float* a = new float[10];
delete[] a;
a[0] = 1.0f;     // the memory is no longer yours
```

**Leak.** Allocating and never freeing. Harmless in a 2-second exercise, fatal
in a training loop that allocates every iteration.

The defence, which you should use from today:

```bash
make debug      # rebuilds with -fsanitize=address,undefined and runs
```

AddressSanitizer catches all four, reports the exact line, and costs you
nothing but a slower binary. When you reach Chapter 18, `compute-sanitizer` is
the same tool for the GPU.

---

## 8. Exercises

### P2.1 - Draw the memory

For this code, draw the boxes-and-arrows picture from section 1 after each
line, with addresses:

```cpp
int x = 10;
int y = 20;
int* p = &x;
int* q = &y;
*p = 30;
p = q;
*p = 40;
```

(a) What are `x` and `y` at the end?
(b) At which line did `p` stop pointing at `x`?
(c) Which line changed `y` without mentioning `y`?
(d) Now write the program and print everything to check. Were you right?

### P2.2 - Swap

(a) Write `void swap(int* a, int* b)` that exchanges two integers.
(b) Call it. Verify.
(c) Write `void broken_swap(int a, int b)` that takes values. Call it. Explain
    precisely why it does nothing, in terms of section 3.
(d) What does `swap(&x, &x)` do? Trace it line by line. Is it safe?

### P2.3 - Pointer arithmetic

```cpp
float a[8] = {1,2,3,4,5,6,7,8};
float* p = a;
```

Predict, then check:

```cpp
*p          *(p + 3)        p[3]        *(a + 5)      a[5]
p += 2;     *p              p[1]        *(p - 1)
```

(e) Print `(void*)p` and `(void*)(p+1)`. What is the numeric difference? Why
    that number?
(f) Now make `p` a `double*` pointing at a `double` array. Same question. What
    general rule did you just confirm?

### P2.4 - Sum a range

(a) Write `float sum(const float* data, int n)`.
(b) Call it on a whole array.
(c) Call it on the second half, without copying: `sum(a + n/2, n - n/2)`. Why
    does that work?
(d) This "pass a pointer into the middle" idea is how every tiled kernel in
    this course addresses its tile. Convince yourself it is the same thing.

### P2.5 - Break it on purpose

Write a program that does each of the four mistakes in section 7. For each:

(a) What happens with `make run`? (Possibly nothing. That is the lesson.)
(b) What happens with `make debug`?
(c) Write down the sanitizer's message for each, so you recognise them later.

This exercise is worth more than the other four together. Bugs you have caused
deliberately are bugs you can diagnose in thirty seconds instead of three
hours.

---

## What you should now believe

- A pointer is a variable containing an address. `&` gets an address, `*`
  follows one.
- Functions copy their arguments, so passing an address is the only way to let
  a function modify or fill something — which is why kernel outputs are
  pointers.
- `a[i]` *is* `*(a + i)`. Arrays are addresses, and pointer arithmetic counts
  in elements.
- C does not know how long your arrays are. Always pass the length.
- Stack is automatic and small; heap is manual and large. `cudaMalloc` is the
  heap, on the other side of the PCIe bus.
- `const float*` marks an input, and in CUDA it is also a performance hint.
- The four classic bugs are silent at runtime and loud under the sanitizers.
  Use `make debug` early.

Next: [P3 - Arrays, indexing and strides](p3-arrays-and-indexing.md). That one
is the most important chapter in this part.
